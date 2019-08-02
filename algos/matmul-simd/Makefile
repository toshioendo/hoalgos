all: AVX512  #default

APP=matmul

AVX2:
	make -f Makefile.AVX2

AVX512:
	make -f Makefile.AVX512


clean:
	rm -f *.o
	rm -f *~
	rm -f core
	rm -f core.*
	rm -f $(APP)
	rm -f a.out
