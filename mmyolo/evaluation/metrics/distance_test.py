# ==============================================================================
# Binaries and/or source for the following packages or projects 
# are presented under one or more of the following open source licenses:
# distance.py    The OpenLane-V2 Dataset Authors    Apache License, Version 2.0
#
# Contact wanghuijie@pjlab.org.cn if you have any issue.
#
# Copyright (c) 2023 The OpenLane-V2 Dataset Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
from similaritymeasures import frechet_dist

# 生成两组包含11个点的随机数据集，并按照 x 方向排序
pred = np.sort(np.random.rand(11, 2), axis=0)
pred2 = pred[::-1] 
gt = np.sort(np.random.rand(11, 2), axis=0)
gt2 = gt[::-1] 

frechet_dist(pred, gt, p=2)
frechet_dist(pred2, gt, p=2)
frechet_dist(pred2, gt2, p=2) #=frechet_dist(pred, gt, p=2)