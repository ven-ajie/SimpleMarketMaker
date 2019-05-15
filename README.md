# market maker-fee chaser
A modified version of https://github.com/deribit/examples

Only for hobby/educational purposes: a copy paste project made by a non-programmer. USE THE SOFTWARE AT YOUR OWN RISK.

Always start by running a trading bot in test server and do not engage money before you understand how it works and what profit/loss you should expect. Do not hesitate to read the source code and understand the mechanism of this bot.

# improvements:
- additional ability to:
  a) respond to strong market movement (both of qty and price)
  b) change bid/ask based on certain parameters (equity, maintenance margin, etc)
  c) perform multi accounts task

- refresh/tidy up/improve efficiency bot code
- add eth qty (x10)
- add/improve bot code related to e mail/time/telegram


# Risk management:
- Prioritise long over short (since short is a bit riskier). Short only for balancing (offsetted margin to avoid liquidation and allow higher long position)
- Focus on protecting capital instead of making transactions
- Cut loss?
- Allow risk position update by e mail/telegram

Basic premises:
- All up, will be go down. And, vice versa
- Profit is earned slowly. Rely on small but continuos 24 H transactions instead of big one time transaction

# Test result:
Tested 3 times in the last 6 months with real money. Actually, can generate decent profit in sideways market. But, finally, all got liquidated. Why?
- Bot not prepared for significant price movement in BTC yet
- Too tight leverage
- Wrong balancing formula/not fully delta neutral yet (possible all long/short in the same time)

