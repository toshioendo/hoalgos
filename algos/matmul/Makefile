all: CPULOOP  #default

APP=matmul

CPULOOP:
	make -f Makefile.CPULOOP

MKL:
	make -f Makefile.MKL


clean:
	rm -f *.o
	rm -f *~
	rm -f core
	rm -f core.*
	rm -f $(APP)
	rm -f a.out
