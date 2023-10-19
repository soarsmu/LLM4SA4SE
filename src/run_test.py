import scipy.stats as stats

macro_y1 = [0.75,	0.72,	0.98,	0.85,	0.59]
macro_y2 = [0.76,	0.72,	1,	0.91,	0.65]

## micro-f1
micro_y1 = [0.83,	0.72,	0.97,	0.86,	0.82]
micro_y2 = [0.84,	0.72,	1,	0.92,	0.83]

statistic, p_value = stats.wilcoxon(macro_y1, macro_y2)

print('macro F1')
print("Wilcoxon Statistic:", statistic)
print("P-Value:", p_value)

alpha = 0.05

if p_value < alpha:
    print("Reject the null hypothesis; there is a significant difference.")
else:
    print("Fail to reject the null hypothesis; there is no significant difference.")

statistic, p_value = stats.wilcoxon(micro_y1, micro_y2)

print('micro F1')
print("Wilcoxon Statistic:", statistic)
print("P-Value:", p_value)

alpha = 0.05

if p_value < alpha:
    print("Reject the null hypothesis; there is a significant difference.")
else:
    print("Fail to reject the null hypothesis; there is no significant difference.")