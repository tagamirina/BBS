CC       = g++
CFLAGS    = -O2 -Wall -std=c++17
#CFLAGS 	 = -O0 -fsanitize=address -fno-omit-frame-pointer -g -Wall -std=c++17

CVINC    = `pkg-config --cflags opencv`
CVLIB    = `pkg-config --libs opencv`
PATHS    = -I/usr/local/include -L/usr/local/lib -I/usr/local/include/opencv

clean :
	rm -f *.exe
	rm -f *.o
	rm -f *.a
	rm -f *.stackdump

bbs : BBS.cpp
	${CC} ${CFLAGS} -fopenmp BBS.cpp ${CVINC} ${CVLIB} -o BBS
	#g++ -O2 -Wall -std=c++17 BBS.cpp `pkg-config --cflags opencv` `pkg-config --libs opencv` -o BBS