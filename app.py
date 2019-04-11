
from collections import OrderedDict
from datetime import datetime
from os.path import getmtime
from time import sleep
from datetime import datetime, timedelta
from utils import (get_logger, print_dict_of_dicts, sort_by_key,
                   ticksize_ceil, ticksize_floor)

import copy as cp
import argparse, logging, math, os, sys, time, traceback
import smtplib

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

		# Me

		imb_bids = ob['bids'][0]['cm_amount']
		imb_asks = ob['asks'][0]['cm_amount']
		imbalance = imb_bids / imb_asks

		# Menentukan arah
		if imbalance > 1:
			imbalance
		else:
			imbalance = (imb_asks / imb_bids) * -1
		# Me

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

		return {'bid': best_bid, 'ask': best_ask, 'imbalance': imbalance, 'imb_bids': imb_bids, 'imb_asks': imb_asks,
		        'bids': bids, 'asks': asks, 'bid_ords': bid_ords, 'ask_ords': ask_ords}

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
		return self.futures[contract]['instrumentName'] [-9:]
	
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

			try:
				MM = self.client.account()['initialMargin'] / self.client.account()['marginBalance']
				avg_price = self.positions[fut]['averagePrice'] * (
						self.positions[fut]['size'] / abs(self.positions[fut]['size']))
				 
			except:
				avg_price = 0
				MM = 0

			spot = self.get_spot()
	
			# Me
			instName = self.get_perpetual (fut)
			imb = self.get_bbo(fut)['imbalance']
			posOpn = sum(OrderedDict({k: self.positions[k]['size'] for k in self.futures.keys()}).values())
			posFut = abs(self.positions[fut]['size'])
			posOB = sum(
				[o['quantity'] for o in [o for o in self.client.getopenorders() if o['direction'] == 'buy']]) - sum(
				[o['quantity'] for o in [o for o in self.client.getopenorders() if o['direction'] == 'sell']])
			posFutBid = sum([o['amount'] for o in [o for o in self.client.positions() if o['direction'] == 'buy']])
			posfutAsk = sum(
				[o['amount'] for o in [o for o in self.client.positions() if o['direction'] == 'sell']])

			PCTAdj = PCT*1 if instName == 'PERPETUAL' else PCT/2 # 1 / 2 = arbitrase aja
			print (PCTAdj)
			NetPosFut=(posFutBid+posfutAsk)/10
			posNet = posOB + posOpn
			Margin = avg_price * (PCTAdj)  
			avg_priceAdj = abs(avg_price) * (PCTAdj)  # up/down
			avg_down = abs(avg_price) - abs(avg_priceAdj)
			avg_up = abs(avg_price) + abs(avg_priceAdj)
			avg_priceAdj2 = abs(avg_price) * PCTAdj  # up/down, mengimbangi kenaikan/penurunan harga, arbitrase aja
			avg_down2 = abs(avg_price) - abs(avg_priceAdj2)
			avg_up2 = abs(avg_price) + abs(avg_priceAdj2)
			nbids = 1
			nasks = 1

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
				bid_mkt = ask_mkt = spot
			elif bid_mkt is None:
				bid_mkt = min(spot, ask_mkt)
			elif ask_mkt is None:
				ask_mkt = max(spot, bid_mkt)
			mid_mkt = 0.25 * (bid_mkt + ask_mkt)

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

				# BIDS
				if place_bids:

					offerOB = bid_mkt

					# cek posisi awal
					if posOpn == 0:
						# posisi baru mulai, order bila bid>ask (memperkecil resiko salah)
						if avg_price == 0 and imb > 0 and posOpn <= 0:
							prc = bid_mkt

						# sudah ada posisi short, buat posisi beli
						elif avg_price < 0 and posFut > 4:
							prc = min(bid_mkt, (abs(avg_price) - abs(Margin)))

						# average down
						elif avg_price > 0 and bid_mkt < avg_price and posFut < 20:
							prc = min(bid_mkt, abs(avg_down))

						else:
							prc = 0

					# net sudah ada posisi, individual masih kosong
					elif avg_price == 0:

						# menyeimbangkan posisi
						if posNet < 0:
							prc = bid_mkt

						else:
							prc = 0

					# sudah ada posisi long
					elif avg_price > 0 :
						# posisi rugi, average down
						if bid_mkt < avg_price and posFut < 20:
							prc = min(bid_mkt, abs(avg_down))

						# posisi laba, kejar balance 0
						elif bid_mkt  >  avg_up2 and NetPosFut <0:
							prc = bid_mkt
							
						# posisi idle, kejar balance <> 0
						elif posFut <= 5:
							prc = bid_mkt

						else:
							prc = 0

					# sudah ada short, ambil laba
					elif avg_price < 0:
						if posFut > 4:
							prc = min(bid_mkt, abs(avg_down))

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
							self.logger.warning('Bid order failed: '
							                    )
				else:
					try:
						self.client.buy(fut, qty, prc, 'true')
					except (SystemExit, KeyboardInterrupt):
						raise
					except Exception as e:
						self.logger.warning('Bid order failed')

				# OFFERS

				if place_asks:

					offerOB = ask_mkt

					# cek posisi awal
					if posOpn == 0:

						# posisi baru mulai, order bila bid>ask (memperkecil resiko salah)
						if avg_price == 0 and imb < 0 and posOpn >= 0:
							prc = bid_mkt

						# sudah ada posisi long, buat posisi jual
						elif avg_price > 0 and posFut > 4:
							prc = max(bid_mkt, (abs(avg_price) + abs(Margin)))

						# average up
						elif avg_price < 0 and bid_mkt > avg_price and posFut <20:
							prc = max(bid_mkt, abs(avg_up))

						else:
							prc = 0

					# net sudah ada posisi, individual masih kosong
					elif avg_price == 0:

						# menyeimbangkan posisi
						if posNet > 0:
							prc = bid_mkt

						else:
							prc = 0

					# sudah ada posisi short
					elif avg_price < 0:

						# posisi rugi, average up
						if bid_mkt > avg_price and posFut < 20:
							prc = max(bid_mkt, abs(avg_up))

						# posisi laba, kejar balance 0
						elif  bid_mkt  < avg_down2 and NetPosFut > 0:
							prc = bid_mkt
							
						# posisi idle, kejar balance <> 0
						elif posFut <= 5:
							prc = bid_mkt
							
						else:
							prc = 0

					# sudah ada long, ambil laba
					elif avg_price > 0:
						if posFut > 4:
							prc = max(bid_mkt, (abs(avg_price) + abs(Margin)))
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
							self.logger.warning('Offer order failed: '
							                    )

				else:
					try:
						self.client.sell(fut, qty, prc, 'true')
					except (SystemExit, KeyboardInterrupt):
						raise
					except Exception as e:
						self.logger.warning('Offer order failed: %s at %s'
						                    % (qty, prc))

			if nbids > len(bid_ords):
				cancel_oids += [o['orderId'] for o in bid_ords[nbids:]]
			if nasks > len(ask_ords):
				cancel_oids += [o['orderId'] for o in ask_ords[nasks:]]
			for oid in cancel_oids:
				try:
					self.client.cancel(oid)
				except:
					self.logger.warning('Order cancellations failed: %s' % oid)
        
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


