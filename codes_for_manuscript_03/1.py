# coding: utf-8
# 开发团队：重大化工LC214 AI小分队
# 开发人员：Tristan
# 开发时间：2023/4/27—21:39
# 文件名称：1 py
# 开发工具：PyCharm

import numpy as np
from lets_plot import *
# %%

# Generate random data-points for the demo.
np.random.seed(12)
data = dict(
    cond=np.repeat(['A', 'B'], 200),
    rating=np.concatenate((np.random.normal(0, 1, 200), np.random.normal(1, 1.5, 200)))
)

# Create plot specification.
p = ggplot(data, aes(x='rating', fill='cond')) + ggsize(500, 250) \
    + geom_density(color='dark_green', alpha=.7) + scale_fill_brewer(type='seq')

# Display plot in 'SciView'.
p.show()