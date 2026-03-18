CC      = cc
CFLAGS  = -std=c11 -Wall -Wextra -O2 -I include
SRCS    = src/utils.c src/dynnet.c src/ode_solver.c src/adjoint.c \
          src/adam.c src/train.c src/spiral.c src/cnf.c src/cnf_train.c \
          src/tests.c src/test_cnf.c src/main.c
TARGET  = neural_ode

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET) -lm

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

.PHONY: run clean
