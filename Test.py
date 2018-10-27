import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
import iexfinance
from datetime import datetime

print([stock['symbol'] for stock in iexfinance.get_available_symbols()])