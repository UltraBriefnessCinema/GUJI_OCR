# 简介
该项目为仿造识典古籍做的本地部署Ai校对古籍，当前某些逻辑欠佳，好在已基本可以进行古籍识别，unicode收字已破10万，OCR模型还需要训练。
仅本人个人使用，很多都未优化。

# 安装部署方式
当前python环境是3.11，可以通过虚拟环境安装。
## 1.NVIDIA CUDA驱动
该项目基于PaddleOCR-GPU版运行，需要大于等于CUDA12.9的toolkit，可以在NVIDIA官网根据自己系统下载后安装，https://developer.nvidia.com/cuda-13-1-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local
我跑在13.1，没有问题。
## 2.在命令行安装paddleOCR-GPU
cd到项目主文件夹。安装paddleocr，
`python -m pip install paddlepaddle-gpu==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu129/`
终端再次运行`pip install -r requirements.txt`
## 3.更改config.py中的路径
模型文件、字体文件，替换为绝对路径。
### 异体字字典
`dic_txt\dict.txt`
用来存放异体字，一行一个字，以空格隔开变体，用于在模型出错时更换，比如改 攺这类形似的字。
`model\type\best.pt`
版面模型，用于分析哪里有字。
`model\slide\best.pt`
切割模型，基于版面分割完的长框来切字。
`fonts`
字体文件夹，包含unicode17所有字。

版面导出和切片导出用于本人优化模型而定，可以不管但必须设置。
### 
## 4.运行主程序
`python main.py`
导入的pdf必须是二值化后的pdf，且界面高度是4000像素，可以使用`compressPDF.py`来处理，os.walk遍历所有子文件夹。
1.右上方有两个按钮，“扫描文件夹推理”，扫描推理yolo模型遍历文件夹中所有子文件夹中的pdf，并在旁边生成json文件
2.“OCR批量处理”，基于已有pdf和同名json时，程序调用gpu对字框识别，在pdf旁边生成同名txt，并按照yolo模型对夹注加个括号，以示区分。
3.以上二者扫描或OCR完后，务必关闭程序再次打开导入pdf。
4.检测页和拆分页对于程序自动排序有问题者，可以手动排序，依次点击字框即可。
5.ocr页面，对于识别有误的字，双击可以手动输入更改，单击可以调出可能的其他字，如遇dict.txt中已写名的异体字，也会出现。
6.ocr页面，对于正确率低于80的字，标红，反之标黑，可以导入校本（即同一本古籍的其他版本文字版），程序如发现有不同，则会标黄，可以“应用校本差异”一键替换修改。
7.点击翻页时，程序会自动保存当前步数。