# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2   --augment none     --logdir exp/cifar10_wrn28-2_16_reference   &> logs/log_0 &
CUDA_VISIBLE_DEVICES='1' python3 -u train.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2   --augment none     --logdir exp/cifar10_wrn28-2_16_reference   &> logs/log_1 &
CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2   --augment none     --logdir exp/cifar10_wrn28-2_16_reference   &> logs/log_2 &
CUDA_VISIBLE_DEVICES='1' python3 -u train.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2   --augment none     --logdir exp/cifar10_wrn28-2_16_reference   &> logs/log_3 &
wait;
CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2   --augment none     --logdir exp/cifar10_wrn28-2_16_reference   &> logs/log_4 &
CUDA_VISIBLE_DEVICES='1' python3 -u train.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2   --augment none     --logdir exp/cifar10_wrn28-2_16_reference   &> logs/log_5 &
CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2   --augment none     --logdir exp/cifar10_wrn28-2_16_reference   &> logs/log_6 &
CUDA_VISIBLE_DEVICES='1' python3 -u train.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2   --augment none     --logdir exp/cifar10_wrn28-2_16_reference   &> logs/log_7 &
wait;
CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2   --augment none     --logdir exp/cifar10_wrn28-2_16_reference   &> logs/log_8 &
CUDA_VISIBLE_DEVICES='1' python3 -u train.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2   --augment none     --logdir exp/cifar10_wrn28-2_16_reference   &> logs/log_9 &
CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2   --augment none     --logdir exp/cifar10_wrn28-2_16_reference   &> logs/log_10 &
CUDA_VISIBLE_DEVICES='1' python3 -u train.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2   --augment none     --logdir exp/cifar10_wrn28-2_16_reference   &> logs/log_11 &
wait;
CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2   --augment none     --logdir exp/cifar10_wrn28-2_16_reference   &> logs/log_12 &
CUDA_VISIBLE_DEVICES='1' python3 -u train.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2   --augment none     --logdir exp/cifar10_wrn28-2_16_reference   &> logs/log_13 &
CUDA_VISIBLE_DEVICES='0' python3 -u train.py --dataset=cifar10 --epochs=100 --save_steps=20 --arch wrn28-2   --augment none     --logdir exp/cifar10_wrn28-2_16_reference   &> logs/log_14 &