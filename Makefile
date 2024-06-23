CFLAGS = -std=c++17 -O2
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXrandr -lXxf86vm -lXi

VulkanTest: main.cpp
	g++ $(CFLAGS) -o VulkanTest main.cpp $(LDFLAGS)

test: VulkanTest
	./VulkanTest

.PHONY: clean

clean:
	rm -f VulkanTest *.o