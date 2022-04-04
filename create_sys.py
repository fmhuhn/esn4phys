import os
import sys
import argparse

import numpy as np

from base.DynamicalSystems import DynamicalSystem

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a dynamical system.')

    parser.add_argument('system_name', metavar='sys_name', type=str,
            help='dynamical system name')

    parser.add_argument('--params', metavar='params', nargs='+', type=float,
            help='dynamical system parameters', default=None)

    parser.add_argument('file_path', metavar='file_path', type=str,
            help='file path of case file')

    args = parser.parse_args()

    System = DynamicalSystem.get_from_name(args.system_name)
    system = System(args.params, args.file_path)
    system.save()