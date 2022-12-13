import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

from fed_api.utils.draw import *

DRAW_BUDGET_CMP = False
DRAW_OTHER_CMP = False
DRAW_CLIENTS_CMP = True
SAMPLE_FREQ = 1
# m_filename = "Fed3/fed3-1104-2022-12-01-22-05-R"
# draw_accuracy(m_filename)
# draw_loss(m_filename)
# draw_time(m_filename)

if DRAW_OTHER_CMP == True:    
    draw_accuracy_cmp(sample_freq=SAMPLE_FREQ)
    # draw_loss_cmp()
    # draw_training_intensity_sum_cmp()
    # draw_time_cmp()
    # draw_goal_cmp()
    draw_accuracy_cmp_with_time(sample_freq=SAMPLE_FREQ)
    # draw_loss_cmp_with_time()
    draw_train_loss_cmp(sample_freq=SAMPLE_FREQ)
    draw_train_loss_cmp_with_time(sample_freq=SAMPLE_FREQ)

# draw_accuracy_budget()
# draw_loss_budget()
# draw_time_budget()

if DRAW_BUDGET_CMP == True:
    x_name = "Budget"
    draw_time_cmp_with_x(x_name, SAMPLE_FREQ)
    # draw_accuracy_cmp_with_budget()
    draw_training_intensity_cmp_with_x(x_name, SAMPLE_FREQ)
    draw_goal_cmp_with_x(x_name, SAMPLE_FREQ)

if DRAW_CLIENTS_CMP == True:
    x_name = "Total numbers of clients"
    draw_time_cmp_with_x(x_name, SAMPLE_FREQ)
    draw_training_intensity_cmp_with_x(x_name, SAMPLE_FREQ)
    draw_goal_cmp_with_x(x_name, SAMPLE_FREQ)
