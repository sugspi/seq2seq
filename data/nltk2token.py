# -*- coding: utf-8 -*-
#
#  Copyright 2015 Pascual Martinez-Gomez
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import logging
import re

data_path = 'snli_0410_formula.txt'#'/home/8/17IA0973/snli_input_data_1214.json'
f = open('snli_0410_token.txt', 'w')
data = []

lines = open(data_path)
for line in lines :
    line = line.split('#')
    l1 = line[0]
    l2 = line[1].rstrip()
    l3 = line[2].rstrip()
    try:
        l1 = re.sub('\(', '( ',l1)
        l1 = re.sub('\)', ' )',l1)
        l1 = re.split('\s|\.', l1)
        strs = ','.join(l1)
        data.append(strs + '#' + l2 + '#' + l3)
    except:
        print(l1)
        continue


for i in data :
    f.write(str(i)+'\n')
f.close()



print(len(data))
