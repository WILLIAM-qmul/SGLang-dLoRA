conda虚拟环境配置时遇到报错
ImportError: /home/cedric/anaconda3/envs/decdiff_env/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /lib/x86_64-linux-gnu/libLLVM-15.so.1)

AI生成项目
1
2
解决方案
检查是否存在

strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX

AI生成项目
1
2
结果如下，我这里是存在version `GLIBCXX_3.4.30’的

GLIBCXX_3.4
GLIBCXX_3.4.1
GLIBCXX_3.4.2
GLIBCXX_3.4.3
GLIBCXX_3.4.4
GLIBCXX_3.4.5
GLIBCXX_3.4.6
GLIBCXX_3.4.7
GLIBCXX_3.4.8
GLIBCXX_3.4.9
GLIBCXX_3.4.10
GLIBCXX_3.4.11
GLIBCXX_3.4.12
GLIBCXX_3.4.13
GLIBCXX_3.4.14
GLIBCXX_3.4.15
GLIBCXX_3.4.16
GLIBCXX_3.4.17
GLIBCXX_3.4.18
GLIBCXX_3.4.19
GLIBCXX_3.4.20
GLIBCXX_3.4.21
GLIBCXX_3.4.22
GLIBCXX_3.4.23
GLIBCXX_3.4.24
GLIBCXX_3.4.25
GLIBCXX_3.4.26
GLIBCXX_3.4.27
GLIBCXX_3.4.28
GLIBCXX_3.4.29
GLIBCXX_3.4.30
GLIBCXX_DEBUG_MESSAGE_LENGTH

AI生成项目

1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
建立软链接

cd /home/cedric/anaconda3/envs/decdiff_env/bin/../lib/
mv libstdc++.so.6 libstdc++.so.6.old
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 libstdc++.so.6

---
我的问题
ImportError: /home/lsl/anaconda3/envs/sglang/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.32' not found (required by /home/lsl/.cache/torch_extensions/py310_cu126/hf3fs_utils/hf3fs_utils.so)

cd /home/lsl/anaconda3/envs/sglang/bin/../lib/
mv libstdc++.so.6 libstdc++.so.6.old
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 libstdc++.so.6