
from collections import OrderedDict
from datetime import datetime
from os.path import getmtime
from time import sleep
from datetime import datetime, timedelta
from utils import (get_logger, print_dict_of_dicts, sort_by_key,
                   ticksize_ceil, ticksize_floor)

import copy as cp
import argparse, logging, math, os, sys, time, traceback

from api import RestClient

KEY = 'AVmfiQceujWF'
SECRET = '33WJ3QOFCBJMUB24OOYANJGWWSVG7RP5'
URL = 'https://test.deribit.com'


# Add command line switches
parser = argparse.ArgumentParser(description='Bot')

# Use production platform/account
parser.add_argument('-p',
                    dest='use_prod',
                    action='store_true')

# Do not display regular status updates to terminal
parser.add_argument('--no-output',
                    dest='output',
                    action='store_false')

# Monitor account only, do not send trades
parser.add_argument('-m',
                    dest='monitor',
                    action='store_true')

# Do not restart bot on errors
parser.add_argument('--no-restart',
                    dest='restart',
                    action='store_false')

args = parser.parse_args()


BP = 1e-4  # one basis point
BTC_SYMBOL = 'btc'
CONTRACT_SIZE = 5  # USD
COV_RETURN_CAP = 100  # cap on variance for vol estimate
DECAY_POS_LIM = 0.1  # position lim decay factor toward expiry
EWMA_WGT_COV = 4  # parameter in % points for EWMA volatility estimate
EWMA_WGT_LOOPTIME = 0.1  # parameter for EWMA looptime estimate
FORECAST_RETURN_CAP = 20  # cap on returns for vol estimate
LOG_LEVEL = logging.INFO
MIN_ORDER_SIZE = 1

MKT_IMPACT = 0.25  # base 1-sided spread between bid/offer
NLAGS = 2  # number of lags in time series
PCT = 100 * BP  # one percentage point
PCT_LIM_LONG = 100  # % position limit long
PCT_LIM_SHORT = 200  # % position limit short
PCT_QTY_BASE = 100  # pct order qty in bps as pct of acct on each order
MIN_LOOP_TIME = 0.2  # Minimum time between loops
RISK_CHARGE_VOL = 0.25  # vol risk charge in bps per 100 vol
SECONDS_IN_DAY = 3600 * 24
SECONDS_IN_YEAR = 365 * SECONDS_IN_DAY
WAVELEN_MTIME_CHK = 15  # time in seconds between check for file change
WAVELEN_OUT = 15  # time in seconds between output to terminal
WAVELEN_TS = 15  # time in seconds between time series update
VOL_PRIOR = 100  # vol estimation starting level in percentage pts

EWMA_WGT_COV *= PCT
MKT_IMPACT *= BP
PCT_LIM_LONG *= PCT
PCT_LIM_SHORT *= PCT
PCT_QTY_BASE *= BP
VOL_PRIOR *= PCT


class MarketMaker(object):

	def __init__(self, monitor=True, output=True):
		self.equity_usd = None
		self.equity_btc = None
		self.equity_usd_init = None
		self.equity_btc_init = None
		self.con_size = float(CONTRACT_SIZE)
		self.client = None
		self.deltas = OrderedDict()
		self.futures = OrderedDict()
		self.futures_prv = OrderedDict()
		self.logger = None
		self.mean_looptime = 1
		self.monitor = monitor
		self.output = output or monitor
		self.positions = OrderedDict()
		self.getsummary = OrderedDict()
		self.spread_data = None
		self.this_mtime = None
		self.ts = None
		self.vols = OrderedDict()

	def create_client(self):
	
		self.client = RestClient(KEY, SECRET, URL)
	
	
	def get_bbo(self, contract):  # Get best b/o excluding own orders

		# Get orderbook
		ob = self.client.getorderbook(contract)
		bids = ob['bids']
		asks = ob['asks']

		ords = self.client.getopenorders(contract)
		bid_ords = [o for o in ords if o['direction'] == 'buy']
		ask_ords = [o for o in ords if o['direction'] == 'sell']
		best_bid = None
		best_ask = None

		last_price = self.client.tradehistory(contract)
		
		try:
			last_price_sell  = [o for o in last_price if o['side'] == 'sell']
			last_price_buy  =  [o for o in last_price if o['side'] == 'buy']
		
		except:
			last_price_sell  = 0
			last_price_buy  =  0

		imb_asks = ob['asks'][0]['cm_amount']
		imb_bids = ob['bids'][0]['cm_amount']
		imbalance = imb_bids / imb_asks

		# Menentukan arah
		if imbalance > 1:
			imbalance
		else:
			imbalance = (imb_asks / imb_bids) * -1


		err = 10 ** -(self.get_precision(contract) + 1)

		for b in bids:
			match_qty = sum([
				o['quantity'] for o in bid_ords
				if math.fabs(b['price'] - o['price']) < err
			])
			if match_qty < b['quantity']:
				best_bid = b['price']
				break

		for a in asks:
			match_qty = sum([
				o['quantity'] for o in ask_ords
				if math.fabs(a['price'] - o['price']) < err
			])
			if match_qty < a['quantity']:
				best_ask = a['price']
				break

		return {'last_price':last_price,'last_price_buy':last_price_buy,
                    'last_price_sell': last_price_sell,'bid': best_bid, 
					'ask': best_ask,'imbalance': imbalance, 'imb_bids': imb_bids, 
					'imb_asks': imb_asks,'bids': bids, 'asks': asks, 
					'bid_ords': bid_ords, 'ask_ords': ask_ords}

	def get_futures(self):  # Get all current futures instruments

		self.futures_prv = cp.deepcopy(self.futures)

		insts = self.client.getinstruments()
		self.futures = sort_by_key({
			i['instrumentName']: i for i in insts if i['kind'] == 'future' 
		})

		for k, v in self.futures.items():
			self.futures[k]['expi_dt'] = datetime.strptime(
				v['expiration'][: -4],
				'%Y-%m-%d %H:%M:%S')

	def get_perpetual (self, contract):
		return self.futures[contract]['instrumentName'] 
	
	def get_spot(self):
		return self.client.index()['btc'] 

	def get_precision(self, contract):
		return self.futures[contract]['pricePrecision']

	def get_ticksize(self, contract):
		return self.futures[contract]['tickSize']

	def output_status(self):

		if not self.output:
			return None

		self.update_status()

		print('')

	def place_orders(self):
		
		if self.monitor:
			return None
	
		for fut in self.futures.keys():
		
		# hold 	= ITEMS ON HAND
		# ord	= ITEMS ON ORDER BOOK
		# net	= LONG + SHORT
		# fut	= INDIVIDUAL ITEM PER INSTRUMENT
		# all	= TOTAL INDIVIDUAL ITEM PER INSTRUMENT PER DIRECTION
			
			instName 		= self.get_perpetual (fut)
			imb 			= self.get_bbo(fut)['imbalance']
			spot 			= self.get_spot()	
			
			##determine various Qty variable 
			hold_fut 		= abs(self.positions[fut]['size'])#individual
			hold_longItem	= len([o['averagePrice'] for o in [o for o in self.client.positions (
										) if o['direction'] == 'buy' and o['currency'] == fut[:3].lower()]])
			hold_shortItem	= len([o['averagePrice'] for o in [o for o in self.client.positions (
										) if o['direction'] == 'sell' and o['currency'] == fut[:3].lower()]])
			hold_longAll 	= sum([o['size'] for o in [o for o in self.client.positions () if o[
								'direction'] == 'buy' and o['currency'] == fut[:3].lower()]])
			hold_shortAll 	= sum([o['size'] for o in [o for o in self.client.positions () if o[
								'direction'] == 'sell' and o['currency'] == fut[:3].lower()]])
			hold_netFut		= (hold_longAll+hold_shortAll)
			
			ord_longFut 	= sum([o['quantity'] for o in [o for o in self.client.getopenorders(
								) if o['direction'] == 'buy' and o['api'] == True and o[
								'instrument'] == fut[:3].lower()]])

			ord_shortFut	= sum([o['quantity'] for o in [o for o in self.client.getopenorders(
								) if o['direction'] == 'sell' and o['api'] == True and o[
								'instrument'] == fut[:3].lower()]])

			bal_btc         = self.client.account()[ 'equity' ]
			qty_lvg			= max(1,round( (bal_btc * spot * 80)/10 * PCT,0)) # 100%-20%

			#determine various price variable
			hold_avgPrcFut 	= self.positions[fut]['averagePrice']*(self.positions[fut]['size'])/abs(
								self.positions[fut]['size']) if self.positions[fut] ['size'] != 0 else 0
			try:
				last_price_buy1 	= self.get_bbo(fut)['last_price_buy'][0] ['price'] 
				last_price_sell1 	= self.get_bbo(fut)['last_price_sell'][0] ['price'] 
				hold_avgPrcShortAll =  sum([o['averagePrice'] for o in [o for o in self.client.positions (
										) if o['direction'] == 'sell' and o['currency'] == fut[:3].lower()]]
										)/len([o['averagePrice'] for o in [o for o in self.client.positions (
										) if o['direction'] == 'sell' and o['currency'] == fut[:3].lower()]]) 

				hold_avgPrcLongAll 	=  sum([o['averagePrice'] for o in [o for o in self.client.positions (
										) if o['direction'] == 'buy' and o['currency'] == fut[:3].lower()]]
										)/len([o['averagePrice'] for o in [o for o in self.client.positions (
										) if o['direction'] == 'buy' and o['currency'] == fut[:3].lower()]])	
							
			except:
				last_price_buy1 	= 0
				last_price_sell1 	= 0
				hold_avgPrcShortAll = 0
				hold_avgPrcLongAll 	= 0

			last_buy 			= last_price_buy1 - (last_price_buy1*PCT*2)
			last_sell 			= abs(last_price_sell1) + abs((last_price_sell1*PCT*2))
			diffperpfut 		= self.client.getsummary(fut)['markPrice'
									]-self.client.getsummary ('BTC-PERPETUAL')['markPrice']
			hold_avgPrcLongPerp	= hold_avgPrcLongAll - (hold_avgPrcLongAll*PCT*10 + diffperpfut)
			hold_avgPrcShortPerp= hold_avgPrcShortAll + (hold_avgPrcShortAll*PCT*10 + diffperpfut)

			avg_down0 			= abs(hold_avgPrcFut) - (abs(hold_avgPrcFut * PCT/2))
			avg_down1 			= abs(hold_avgPrcFut) - (abs(hold_avgPrcFut * PCT))
			avg_down2 			= abs(hold_avgPrcFut) - (abs(hold_avgPrcFut * PCT * 2))
			avg_down5 			= abs(hold_avgPrcFut) - (abs(hold_avgPrcFut * PCT * 5))
			avg_down10 			= abs(hold_avgPrcFut) - (abs(hold_avgPrcFut * PCT * 10))
			avg_down20 			= abs(hold_avgPrcFut) - (abs(hold_avgPrcFut * PCT * 20))
			avg_up0 			= abs(hold_avgPrcFut) + (abs(hold_avgPrcFut) * PCT/2)
			avg_up1	 			= abs(hold_avgPrcFut) + (abs(hold_avgPrcFut) * PCT)
			avg_up2 			= abs(hold_avgPrcFut) + (abs(hold_avgPrcFut) * PCT * 2)
			avg_up5 			= abs(hold_avgPrcFut) + (abs(hold_avgPrcFut) * PCT * 5)
			avg_up10 			= abs(hold_avgPrcFut) + (abs(hold_avgPrcFut) * PCT * 10)
			avg_up20 			= abs(hold_avgPrcFut) + (abs(hold_avgPrcFut) * PCT * 20)

			#Menghitung kuantitas beli/jual
			# maks kuantitas by maks leverage
			
			nbids 				= 1
			nasks 				= 1

			place_bids = 'true'
			place_asks = 'true'

			if not place_bids and not place_asks:
				print('No bid no offer for %s' % fut)
				continue

			tsz = self.get_ticksize(fut)
			
			# Perform pricing
			vol = max(self.vols[BTC_SYMBOL], self.vols[fut])

			eps = BP * vol * RISK_CHARGE_VOL
			riskfac = math.exp(eps)

			bbo = self.get_bbo(fut)

			bid_mkt = bbo['bid']
			ask_mkt = bbo['ask']
			
			if bid_mkt is None and ask_mkt is None:
				bid_mkt = ask_mkt 
			elif bid_mkt is None:
				bid_mkt =  ask_mkt
			elif ask_mkt is None:
				ask_mkt = mbid_mkt
			mid_mkt = self.get_ticksize(fut) * (bid_mkt + ask_mkt)

			ords = self.client.getopenorders(fut)
			cancel_oids = []
			bid_ords = ask_ords = []

			if place_bids:
				bid_ords = [o for o in ords if o['direction'] == 'buy']
				len_bid_ords = min(len(bid_ords), nbids)
				bid0 = mid_mkt * math.exp(-MKT_IMPACT)

				bids = [bid0 * riskfac ** -i for i in range(1, nbids + 1)]

				bids[0] = ticksize_floor(bids[0], tsz)

			if place_asks:
				ask_ords = [o for o in ords if o['direction'] == 'sell']
				len_ask_ords = min(len(ask_ords), nasks)
				ask0 = mid_mkt * math.exp(MKT_IMPACT)

				asks = [ask0 * riskfac ** i for i in range(1, nasks + 1)]

				asks[0] = ticksize_ceil(asks[0], tsz)


			for i in range(max(nbids, nasks)):
#hold_longItem hold_shortItem
				# BIDS
				if place_bids:
					
					offerOB = bid_mkt

					if hold_avgPrcFut == 0 and imb > 0 and abs(ord_longFut) < qty_lvg and abs(hold_netFut)<2 :
						#and abs(hold_longAll/max(1,hold_shortAll)) < 2
						# posisi baru mulai, order bila bid>ask 
						if instName [-10:] != '-PERPETUAL' and hold_longAll <= 0 and ord_longFut ==0 :
							prc = bid_mkt
						
						elif instName [-10:] == '-PERPETUAL' and hold_avgPrcLongAll !=0 :
							prc = min(bid_mkt,hold_avgPrcLongPerp)

						else:
							prc = 0
						
					# sudah ada short, ambil laba
					
					elif hold_avgPrcFut < 0 and hold_avgPrcFut != 0:
						prc = min(bid_mkt, abs(avg_down0))

					# average down pada harga < 5%, 10% & 20%
					elif bid_mkt < hold_avgPrcFut and hold_avgPrcFut != 0 and abs(ord_longFut) < 1 and hold_shortAll !=0 :
						#and abs(hold_longAll/max(1,hold_shortAll)) < 2
						if  hold_longAll <= qty_lvg :
							prc = min(bid_mkt, abs(avg_down2
								), last_buy) if last_buy != 0 else min(
									bid_mkt, abs(avg_down2))
							
						elif  hold_longAll < qty_lvg * 3 and instName [-10:] != '-PERPETUAL':
							prc = min(bid_mkt, abs(avg_down20
								), last_buy) if last_buy != 0 else min(
									bid_mkt, abs(avg_down20))

						elif  hold_longAll < qty_lvg * 4 and instName [-10:] == '-PERPETUAL':
							prc = min(bid_mkt, abs(avg_down20
								), last_buy) if last_buy != 0 else min(
									bid_mkt, abs(avg_down20))
			
						else:
							prc = 0

					else:
						prc = 0
					
				else:
					prc = 0

				qty = 1
					
				if i < len_bid_ords:

					oid = bid_ords[i]['orderId']
					try:
						self.client.edit(oid, qty, prc)
					except (SystemExit, KeyboardInterrupt):
						raise
					except:
						try:
							self.client.buy(fut, qty, prc, 'true')
							cancel_oids.append(oid)
							self.logger.warning('Edit failed for %s' % oid)
						except (SystemExit, KeyboardInterrupt):
							raise
						except Exception as e:
							self.logger.warning('Bid order failed: %s'% instName
							                    )
				else:
					try:
						self.client.buy(fut, qty, prc, 'true')
					except (SystemExit, KeyboardInterrupt):
						raise
					except Exception as e:
						self.logger.warning('Bid order failed %s'% instName)

				# OFFERS

				if place_asks:
				
					offerOB = ask_mkt
					# cek posisi awal
					if hold_avgPrcFut == 0 and imb <  0 and abs(ord_shortFut) < qty_lvg and abs(hold_netFut)<2:
#and abs(hold_shortAll/max (1,hold_longAll)) <2
						# posisi baru mulai, order bila bid<ask (memperkecil resiko salah)
						if instName [-10:] != '-PERPETUAL' and abs(hold_shortAll) <= 0 and ord_shortFut ==0 :
							prc = bid_mkt
							
						elif instName [-10:] == '-PERPETUAL' and hold_avgPrcShortAll !=0:
							prc =  max(bid_mkt,hold_avgPrcShortPerp)
								
						else:
							prc = 0
					# sudah ada long, ambil laba
					
					elif hold_avgPrcFut > 0 and hold_avgPrcFut != 0 :
						prc = max(bid_mkt, abs(avg_up0))
						
					# average up pada harga < 5%, 10% & 20%
					elif bid_mkt > hold_avgPrcFut and hold_avgPrcFut != 0 and abs(
						ord_shortFut) < 1 and hold_longAll !=0 :
					#and abs(hold_shortAll/max (1,hold_longAll))
						if abs(hold_shortAll) <= qty_lvg:
							prc = max(bid_mkt, abs(avg_up2
									), last_sell) if last_sell != 0 else max(
									bid_mkt, abs(avg_up2))
						
						elif abs(hold_shortAll) < qty_lvg * 3 and instName [-10:] != '-PERPETUAL':
							prc = max(bid_mkt, abs(avg_up20
									), last_sell) if last_sell != 0 else max(
									bid_mkt, abs(avg_up20))
						
						elif abs(hold_shortAll) < qty_lvg * 4 and instName [-10:] == '-PERPETUAL':
							prc = max(bid_mkt, abs(avg_up20
									), last_sell) if last_sell != 0 else max(
									bid_mkt, abs(avg_up20))
						
						else:
							prc = 0
				
					else:
						prc = 0
				
				else:
					prc = 0

				qty = 1

				if i < len_ask_ords:
					oid = ask_ords[i]['orderId']
					try:
						self.client.edit(oid, qty, prc)
					except (SystemExit, KeyboardInterrupt):
						raise
					except:
						try:
							self.client.sell(fut, qty, prc, 'true')
							cancel_oids.append(oid)
							self.logger.warning('Sell Edit failed for %s' % oid)
						except (SystemExit, KeyboardInterrupt):
							raise
						except Exception as e:
							self.logger.warning('Offer order failed: %s'% instName
							                    )

				else:
					try:
						self.client.sell(fut, qty, prc, 'true')
					except (SystemExit, KeyboardInterrupt):
						raise
					except Exception as e:
						self.logger.warning('Offer order failed: %s'% instName
						                    )

			if nbids > len(bid_ords):
				cancel_oids += [o['orderId'] for o in bid_ords[nbids:]]
			if nasks > len(ask_ords):
				cancel_oids += [o['orderId'] for o in ask_ords[nasks:]]
				
		
			for oid in cancel_oids:
				try:
					self.client.cancel(oid)
				except:
					self.logger.warning('Order cancellations failed: %s' % oid)

			#cancell all orders when: any  executions on order book, item quantity outstanding on orderbook> 1, or > 10 seconds
			#
			try:
				ord_diffTime 	= (self.client.gettime()/1000) - (ords [0] ['created']/1000)
				hold_diffTime 	= (self.client.gettime()/1000)-(self.get_bbo(fut)['last_price'][0] ['timeStamp']/1000)
		
			except:
				ord_diffTime 	= 9
				hold_diffTime	= 2

			if hold_diffTime < 1 or ord_shortFut >2 or ord_longFut>2 or ord_diffTime > 10:
				while True:
					self.client.cancelall()
					sleep (10)
					break   		

	def restart(self):
		try:
			strMsg = 'RESTARTING'
			print(strMsg)
			self.client.cancelall()
			strMsg += ' '
			for i in range(0, 5):
				strMsg += '.'
				print(strMsg)
				sleep(1)
		except:
			pass
		finally:
			os.execv(sys.executable, [sys.executable] + sys.argv)

	def run(self):

		self.run_first()
		self.output_status()

		t_ts = t_out = t_loop = t_mtime = datetime.utcnow()

				
		while True:

			self.get_futures()
			
			# Restart if a new contract is listed
			if len(self.futures) != len(self.futures_prv):
				self.restart()

			self.update_positions()

			t_now = datetime.utcnow()

			# Update time series and vols
			if (t_now - t_ts).total_seconds() >= WAVELEN_TS:
				t_ts = t_now
				self.update_timeseries()
				self.update_vols()

			self.place_orders()

			# Display status to terminal
			if self.output:
				t_now = datetime.utcnow()
				if (t_now - t_out).total_seconds() >= WAVELEN_OUT:
					self.output_status();
					t_out = t_now

			# Restart if file change detected
			t_now = datetime.utcnow()
			if (t_now - t_mtime).total_seconds() > WAVELEN_MTIME_CHK:
				t_mtime = t_now
				if getmtime(__file__) > self.this_mtime:
					self.restart()

			t_now = datetime.utcnow()
			looptime = (t_now - t_loop).total_seconds()

			# Estimate mean looptime
			w1 = EWMA_WGT_LOOPTIME
			w2 = 1.0 - w1
			t1 = looptime
			t2 = self.mean_looptime

			self.mean_looptime = w1 * t1 + w2 * t2

			t_loop = t_now
			sleep_time = MIN_LOOP_TIME - looptime
			if sleep_time > 0:
				time.sleep(sleep_time)
			if self.monitor:
				time.sleep(WAVELEN_OUT)

	def run_first(self):

		self.create_client()
		self.client.cancelall()
		self.logger = get_logger('root', LOG_LEVEL)

		# Get all futures contracts
		self.get_futures()
		self.this_mtime = getmtime(__file__)
		self.symbols = [BTC_SYMBOL] + list(self.futures.keys());
		self.symbols.sort()
		self.deltas = OrderedDict({s: None for s in self.symbols})

		# Create historical time series data for estimating vol
		ts_keys = self.symbols + ['timestamp'];
		ts_keys.sort()

		self.ts = [
			OrderedDict({f: None for f in ts_keys}) for i in range(NLAGS + 1)
		]

		self.vols = OrderedDict({s: VOL_PRIOR for s in self.symbols})

		self.start_time = datetime.utcnow()
		self.update_status()
		self.equity_usd_init = self.equity_usd
		self.equity_btc_init = self.equity_btc

	def update_status(self):

		account = self.client.account()
		spot = self.get_spot()

	def update_positions(self):

		self.positions = OrderedDict({f: {
			'size': 0,
			'sizeBtc': 0,
			'indexPrice': None,
			'markPrice': None
		} for f in self.futures.keys()})
		positions = self.client.positions()

		for pos in positions:
			if pos['instrument'] in self.futures:
				self.positions[pos['instrument']] = pos

	def update_timeseries(self):

		if self.monitor:
			return None

		for t in range(NLAGS, 0, -1):
			self.ts[t] = cp.deepcopy(self.ts[t - 1])

		spot = self.get_spot()
		self.ts[0][BTC_SYMBOL] = spot

		for c in self.futures.keys():

			bbo = self.get_bbo(c)
			bid = bbo['bid']
			ask = bbo['ask']

			if not bid is None and not ask is None:
				mid = 0.5 * (bbo['bid'] + bbo['ask'])
			else:
				continue
			self.ts[0][c] = mid

		self.ts[0]['timestamp'] = datetime.utcnow()

	def update_vols(self):

		if self.monitor:
			return None

		w = EWMA_WGT_COV
		ts = self.ts

		t = [ts[i]['timestamp'] for i in range(NLAGS + 1)]
		p = {c: None for c in self.vols.keys()}
		for c in ts[0].keys():
			p[c] = [ts[i][c] for i in range(NLAGS + 1)]

		if any(x is None for x in t):
			return None
		for c in self.vols.keys():
			if any(x is None for x in p[c]):
				return None

		NSECS = SECONDS_IN_YEAR
		cov_cap = COV_RETURN_CAP / NSECS

		for s in self.vols.keys():
			x = p[s]
			dx = x[0] / x[1] - 1
			dt = (t[0] - t[1]).total_seconds()
			v = min(dx ** 2 / dt, cov_cap) * NSECS
			v = w * v + (1 - w) * self.vols[s] ** 2

			self.vols[s] = math.sqrt(v)


if __name__ == '__main__':

	try:
		mmbot = MarketMaker(monitor=args.monitor, output=args.output)
		mmbot.run()
	except(KeyboardInterrupt, SystemExit):
		print("Cancelling open orders")
		mmbot.client.cancelall()
		sys.exit()
	except:
		print(traceback.format_exc())
		if args.restart:
			mmbot.restart()

