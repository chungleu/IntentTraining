# *****************************************************************
#
# Licensed Materials - Property of IBM
#
# IBM Watson Conversation Test Suite
# (C) Copyright IBM Corp. 2017. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
#
# *****************************************************************

from os.path import dirname, abspath, join
import sys

sys.path.append(join(dirname(dirname(abspath(__file__)))))

DATA_FOLDER = join(dirname(abspath(__file__)), 'data')
OUTPUT_FOLDER = join(dirname(abspath(__file__)), 'output')