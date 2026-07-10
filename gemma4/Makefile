CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -std=c11
LDLIBS = -lm -lpthread

OBJS = g4_quants.o g4_dequant.o g4_gguf.o g4_kvstore.o g4_tokenizer.o g4.o

all: g4_test g4_toy_test g4_tok_test g4_session_test

%.o: %.c g4.h g4_quants.h g4_gguf.h g4_kvstore.h g4_tokenizer.h
	$(CC) $(CFLAGS) -c -o $@ $<

g4_test: tests/g4_test.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ tests/g4_test.c $(OBJS) $(LDLIBS)

g4_toy_test: tests/g4_toy_test.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ tests/g4_toy_test.c $(OBJS) $(LDLIBS)

g4_tok_test: tests/g4_tok_test.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ tests/g4_tok_test.c $(OBJS) $(LDLIBS)

g4_session_test: tests/g4_session_test.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ tests/g4_session_test.c $(OBJS) $(LDLIBS)

test: g4_test g4_toy_test g4_tok_test g4_session_test
	./g4_test
	@if [ -f tests/vectors/toy.gguf ]; then ./g4_toy_test; else echo "toy.gguf missing, skipping M1 test"; fi
	@if [ -f tests/vectors/toy.gguf ]; then ./g4_session_test; else echo "toy.gguf missing, skipping session test"; fi
	@if [ -f tests/vectors/tok_table.bin ]; then ./g4_tok_test; else echo "tok_table.bin missing (scripts/gen_tokenizer_vectors.py), skipping tokenizer test"; fi

clean:
	rm -f g4_test g4_toy_test g4_tok_test g4_session_test *.o

.PHONY: all test clean
