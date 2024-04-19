TARGET = neuralnetwork.app
CC = g++
CFLAGS = -std=gnu++20 -Wall
SRCDIR = src
INCDIR = include

BOOST_DIR = /usr/local/Cellar/boost
BOOST_LIBS = -lboost_serialization
BOOST_INCLUDE = -I$(BOOST_DIR)/include
BOOST_LDFLAGS = -L$(BOOST_DIR)/lib -Wl,-rpath -Wl,$(BOOST_DIR)/lib

SRCS = $(wildcard $(SRCDIR)/*.cpp)
OBJS = $(SRCS:.cpp=.o)

#BOOST_LDFLAGS = /usr/local/Cellar/boost/lib

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(BOOST_LDFLAGS) $(OBJS) -o $(TARGET) $(BOOST_LIBS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@ -I$(INCDIR)

.PHONY: clean
clean:
	rm -f $(OBJS) $(TARGET)
