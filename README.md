# Market maker-fee chaser
Made by a non-programmer, only for hobby/educational purposes (a semi-copy paste project from https://github.com/deribit/examples). USE THE SOFTWARE AT YOUR OWN RISK.

Always start by running a trading bot in test server and do not engage money before you understand how it works and what profit/loss you should expect. Do not hesitate to read the source code and understand the mechanism of this bot.

# Improvements:
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
- Place right position from the beginning
- Cut loss?
- Allow risk position update by e mail/telegram
