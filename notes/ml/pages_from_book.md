**Discrete Probability Distributions: Summary.**

1. To summarize, we’ve seen that **Discrete Probability Distributions** can be derived from histograms. And while these can be useful, they require a lot of data that can be expensive and time‐consuming to get, and it’s not always clear what to do about the blank spaces.

2. So, we usually use **mathematical equations**, like the equation for the **Binomial Distribution**, instead:

\[
p(x \mid n, p) = \frac{n!}{x!(n - x)!}\,p^x\,(1 - p)^{n - x}.
\]

The Binomial Distribution is useful for anything that has **binary outcomes** (wins and losses, yeses and noes, etc.), but there are lots of other Discrete Probability Distributions.

3. For example, when we have events that happen in discrete units of time or space—like reading **10 pages an hour**—we can use the **Poisson Distribution**:

\[
p(x \mid \lambda) = \frac{e^{-\lambda}\,\lambda^x}{x!}.
\]

4. There are lots of other **Discrete Probability Distributions** for lots of other types of data. In general, their equations look intimidating, but looks can be deceiving. Once you know what each symbol means, you just plug in the numbers and do the math. **BAM!!!**

Now let’s talk about **Continuous Probability Distributions.**