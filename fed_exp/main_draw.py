import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

from fed_api.utils.draw import *

# m_filename = "Fed3/fed3-1104-2022-12-01-22-05-R"
# draw_accuracy(m_filename)
# draw_loss(m_filename)
# draw_time(m_filename)

draw_accuracy_cmp()
draw_loss_cmp()
draw_training_intensity_sum_cmp()
draw_time_cmp()
draw_goal_cmp()

# draw_accuracy_budget()
# draw_loss_budget()
# draw_time_budget()

draw_accuracy_cmp_with_time()
draw_loss_cmp_with_time()
# draw_time_cmp_with_budget()
# draw_accuracy_cmp_with_budget()
# draw_goal_cmp_with_budget()