import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

data = pd.read_excel('.../data.xlsx')

#Question 1a)
mu = 14.7
sigma = 33
rand_data = np.random.normal(mu, sigma, 10000)
rand_data.sort()

neg_return = 0
y_plot = stats.norm.pdf(rand_data, mu, sigma)
plt.plot(rand_data, y_plot)
plt.fill_between(np.linspace(min(rand_data),neg_return,5000), 
                 stats.norm.pdf(np.linspace(min(rand_data),neg_return,5000), mu, sigma))
plt.show()

p_a = stats.norm.cdf(neg_return, mu, sigma) 
print(f'Approximately, {p_a*100:.1f}% of years this portfolio will lose money.')

#Question 1b)
top_percent = 0.15
remainder = 1 - top_percent

p_b = stats.norm.ppf(remainder, mu, sigma)
print(f'The cutoff for the highest 15% of annual returns is {p_b:.1f}%.')

plt.plot(rand_data, y_plot)
plt.fill_between(np.linspace(p_b,max(rand_data),5000), 
                 stats.norm.pdf(np.linspace(p_b,max(rand_data),5000), mu, sigma))
plt.show()

#%%
# Question 2a)
alpha = 0.1
mu0 = 20
mu1 = 14
sigma = 6
sample = 20

t_crit = stats.t.ppf(1 - alpha, df=sample-1)

t_stat = (mu1-mu0) / (sigma/math.sqrt(sample))

p = stats.t.cdf(mu1,sample-1,loc=mu0,scale=(sigma/math.sqrt(sample))) 

if p < alpha:
    print(f'Since the p-value {p:.4f} is less than 0.1, we can reject the null hypothesis and\
          confirm the reward system was effective.')
else:
    print(f'Since the p-value {p:.4f} is not less than 0.1, we cannot reject the null hypothesis.')

    
#Question 2b)
mu1 = 16
crit = stats.norm.cdf(1-alpha, mu0, sigma)
power = stats.norm.cdf(crit, mu1, sigma)
beta = 1 - power

print(f'The probability the new reward system reduced absenteeism and we missed\
      it is {beta*100:.2f}%')

#Question 2c)
sample = 20
while True:
    sample += 1
    t_stat = (mu0-mu1) / (sigma/math.sqrt(sample))
    power = stats.norm.cdf(t_stat, mu1, sigma)
    if power >= 0.95:
        print(f'The sample size should be {sample} and the power is {power*100:.1f}%')
        break

#%%
#Question 3a)
data.head()

days = len(data.columns) - 2
prob = 0.532
expected_values = [(((1-prob)**f) * prob) * data.iloc[0,-1] if f != range(days)[-1] 
                   else ((1-prob)**f) * data.iloc[0,-1] for f in range(days)]

print(expected_values)

#Question 3b)
obs = data.iloc[0,1:-1].to_numpy()
chi_stat, p = stats.chisquare(obs, f_exp=expected_values)
critical = stats.chi2.ppf(0.95, 6)

test_stat = []
for i in range(7):
    test_stat.append((obs[i]-expected_values[i])**2/expected_values[i])
test_stat = sum(np.array(test_stat))

print(f'The calculated critical value is {critical:.2f}')
print(f'The chi-squared test stat is {chi_stat:.2f} and the p-value is {p:.2f}')