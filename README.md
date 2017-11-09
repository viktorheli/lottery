# lottery
This is a joke project. But in any joke there is truth. :)
Goal of this project - check popular lottery 6 from 45. My hypothesis was that if this lottery is scam, then results of lottery will be predictable. I collected data from 3000 editions of the lottery, train neural network, and check it on next results of lottery. But in this experiment a don't found any correlations in results.

For training we need dataset with results of lottery in csv format. In other csv file fill last results  of lottery. Important: fill blank fields with 0 (zeroes). Zeroes in fields needs because i did not invented how do this in lua. Openoffice does this task more fast then me. :)

Download dataset: https://www.dropbox.com/s/67jh1ez9ntfsjy5/data6from45.csv?dl=0

Download example input data for test: https://www.dropbox.com/s/0gj11okxdlqzlt9/6from45input.csv?dl=0

Put this csv files in same directory with lua files. 

How to use:

1. Make own dataset or download from link above in csv format. Create dataset in t7 format with make-dataset.lua

2. Train network with train.lua . In my case network need 500000 train iteration. With cuda - ~1h on GTX 1070

3. Make input data in csv format. Put in csv file last results from lottery 6 from 45. 

4. Do predict lottery result with predict.lua

5. PROFIT :)


NOTE: 
starting all scripts with -h key will display short help

Good luck.
