TARGET = myApp
CC = g++
CFLAGS = -std=gnu++20 -Wall -Wextra -O2
SRCDIR = src
INCDIR = include
LIBDIR = lib

BOOST_DIR = /usr/local/Cellar/boost
BOOST_LIBS = -lboost_serialization
BOOST_INCLUDE = -I$(BOOST_DIR)/include
BOOST_LDFLAGS = -L$(BOOST_DIR)/lib -Wl,-rpath -Wl,$(BOOST_DIR)/lib

SRCS = $(wildcard $(SRCDIR)/*.cpp)
OBJS = $(SRCS:.cpp=.o)

LIBTARGET = $(LIBDIR)/neuralnetwork.a
LIBSRCS = $(filter-out $(SRCDIR)/main.cpp, $(SRCS))
LIBOBJS = $(LIBSRCS:.cpp=.o)

all: $(TARGET) $(LIBTARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(BOOST_LDFLAGS) $(OBJS) -o $(TARGET) $(BOOST_LIBS)

$(LIBTARGET): $(LIBOBJS) | $(LIBDIR)
	ar rcs $(LIBTARGET) $(LIBOBJS)

$(LIBDIR):
	mkdir -p $(LIBDIR)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@ -I$(INCDIR)

.PHONY: clean
clean:
	rm -f $(OBJS) $(LIBOBJS) $(TARGET) $(LIBTARGET)
