#!/bin/bash

for ((i=1 ; i<=100 ; i++))
do
if [ "$i" -ge 100 ]

then
	./BBS -gamma 2 -pz 3 -tmp ./MainData/tmp001.png -i ./MainData/img${i}.png -txt ./MainData/img${i}.txt -res ./IMG/output${i}.png -log ./TXT/output${i}.txt -logT ./IMG/logT${i}.png -logI ./IMG/logI${i}.png -v 1 -mode 1
else

	if [ "$i" -ge 10 ]

	then
		./BBS -gamma 2 -pz 3 -tmp ./MainData/tmp001.png -i ./MainData/img0${i}.png -txt ./MainData/img0${i}.txt -res ./IMG/output${i}.png -log ./TXT/output${i}.txt -logT ./IMG/logT${i}.png -logI ./IMG/logI${i}.png -v 1 -mode 1
	else
		./BBS -gamma 2 -pz 3 -tmp ./MainData/tmp001.png -i ./MainData/img00${i}.png -txt ./MainData/img00${i}.txt -res ./IMG/output${i}.png -log ./TXT/output${i}.txt -logT ./IMG/logT${i}.png -logI ./IMG/logI${i}.png -v 1 -mode 1
	fi
fi
done