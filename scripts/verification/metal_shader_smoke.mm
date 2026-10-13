#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    std::ifstream input(argv[1]);
    if (!input) return 2;
    std::string hostCode((std::istreambuf_iterator<char>(input)),
                         std::istreambuf_iterator<char>());
    const std::string startTag = "R\"METAL(";
    const std::string endTag = ")METAL\"";
    size_t start = hostCode.find(startTag);
    if (start == std::string::npos) return 2;
    start += startTag.size();
    size_t end = hostCode.find(endTag, start);
    if (end == std::string::npos) return 2;

    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "No Metal device is available\n";
            return 3;
        }
        std::string shader = hostCode.substr(start, end - start);
        NSError* error = nil;
        NSString* source = [NSString stringWithUTF8String:shader.c_str()];
        id<MTLLibrary> library = [device newLibraryWithSource:source
                                                      options:nil error:&error];
        if (!library) {
            std::cerr << "Shader compilation failed: "
                      << [[error localizedDescription] UTF8String] << "\n";
            return 1;
        }
        for (NSString* name in @[@"volt_update", @"curr_update"]) {
            id<MTLFunction> function = [library newFunctionWithName:name];
            if (!function) return 1;
            id<MTLComputePipelineState> pipeline =
                [device newComputePipelineStateWithFunction:function error:&error];
            if (!pipeline) {
                std::cerr << "Pipeline creation failed: "
                          << [[error localizedDescription] UTF8String] << "\n";
                return 1;
            }
        }
        std::cout << "PASS: Metal shader compilation and both compute pipelines\n";
    }
    return 0;
}
