CFLAGS = -std=c++17 -O2 -I$(STB_INCLUDE_PATH)
LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXrandr -lXxf86vm -lXi

STB_INCLUDE_PATH = .

VulkanTest: main.cpp
	g++ $(CFLAGS) -o VulkanTest main.cpp $(LDFLAGS)

test: VulkanTest
	./VulkanTest

.PHONY: clean

clean:
	rm -f VulkanTest *.o