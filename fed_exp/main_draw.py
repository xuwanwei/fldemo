import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

from utils.args_parser import add_args_for_drawing
from fed_api.utils.draw import *

DRAW_BUDGET_CMP = True
DRAW_OTHER_CMP = False
DRAW_CLIENTS_CMP = False
SAMPLE_FREQ = 1
# m_filename = "Fed3/fed3-1104-2022-12-01-22-05-R"
# draw_accuracy(m_filenam)
# draw_loss(m_filename)
# draw_time(m_filename)

if __name__ == "__main__":
    parser = add_args_for_drawing(argparse.ArgumentParser(description=
    'drawing'))
    args = parser.parse_args()
    if args.type == 'acc':    
        draw_accuracy_cmp(args)
        draw_accuracy_cmp_with_time(args)
    elif args.type == 'loss':
        draw_train_loss_cmp(args)
        draw_train_loss_cmp_with_time(args)
    elif args.type == 'budget':
        if args.xlabel == 'undefined':
            args.xlabel = "Budget"
        draw_time_cmp_with_x(args)
        # draw_accuracy_cmp_with_budget()
        draw_training_intensity_cmp_with_x(args)
        draw_goal_cmp_with_x(args)
    elif args.type == 'client_nums':
        if args.xlabel == 'undefined':
            args.xlabel = "Total numbers of clients"
        draw_time_cmp_with_x(args)
        draw_training_intensity_cmp_with_x(args)
        draw_goal_cmp_with_x(args)
    # dif Budget T-A, T-L
    elif args.type == 'B-T-AL':
        draw_accuracy_cmp_para(args)
        draw_accuracy_cmp_with_time_para(args)
        draw_train_loss_cmp_para(args)
        draw_train_loss_cmp_with_time_para(args)
