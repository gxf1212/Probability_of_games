import math

def binomial_prob_gte(k, n, p):
    """
    计算二项分布 B(n, p) 中，结果大于等于 k 的概率。
    P(X >= k) = sum_{i=k to n} C(n, i) * p^i * (1-p)^(n-i)
    """
    if k < 0:
        return 1.0
    prob = 0.0
    for i in range(k, n + 1):
        try:
            prob += math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
        except (ValueError, OverflowError):
            # 处理浮点数精度问题或数值过大问题
            continue
    return prob

