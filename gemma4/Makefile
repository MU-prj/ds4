CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -std=c11
LDLIBS = -lm -lpthread

OBJS = g4_quants.o g4_dequant.o g4_gguf.o g4_kvstore.o

all: g4_test

%.o: %.c g4_quants.h g4_gguf.h g4_kvstore.h
	$(CC) $(CFLAGS) -c -o $@ $<

g4_test: tests/g4_test.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ tests/g4_test.c $(OBJS) $(LDLIBS)

test: g4_test
	./g4_test

clean:
	rm -f g4_test *.o

.PHONY: all test clean
