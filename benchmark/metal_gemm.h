#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#import "gemm.h"

template <class T> class MetalGEMM : public GEMM<T>
{
protected:
    size_t n;
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    MPSMatrixDescriptor *descA;
    MPSMatrixDescriptor *descB;
    MPSMatrixDescriptor *descC;
    id<MTLBuffer> bufferA;
    id<MTLBuffer> bufferB;
    id<MTLBuffer> bufferC;
    MPSMatrix *matA;
    MPSMatrix *matB;
    MPSMatrix *matC;
    MPSMatrixMultiplication *kernel;

public:
    MetalGEMM(size_t n) : n(n)
    {
        device = MTLCreateSystemDefaultDevice();
        commandQueue = [device newCommandQueue];

        descA =
            [MPSMatrixDescriptor matrixDescriptorWithRows:n
                                                  columns:n
                                                 rowBytes:sizeof(float) * n
                                                 dataType:MPSDataTypeFloat32];
        descB =
            [MPSMatrixDescriptor matrixDescriptorWithRows:n
                                                  columns:n
                                                 rowBytes:sizeof(float) * n
                                                 dataType:MPSDataTypeFloat32];
        descC =
            [MPSMatrixDescriptor matrixDescriptorWithRows:n
                                                  columns:n
                                                 rowBytes:sizeof(float) * n
                                                 dataType:MPSDataTypeFloat32];

        bufferA = [device newBufferWithLength:[descA matrixBytes]
                                      options:MTLResourceStorageModeShared];
        bufferB = [device newBufferWithLength:[descA matrixBytes]
                                      options:MTLResourceStorageModeShared];
        bufferC = [device newBufferWithLength:[descA matrixBytes]
                                      options:MTLResourceStorageModeShared];

        matA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
        matB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
        matC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];

        kernel = [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                   transposeLeft:false
                                                  transposeRight:false
                                                      resultRows:n
                                                   resultColumns:n
                                                 interiorColumns:n
                                                           alpha:1.0
                                                            beta:0.0];
    }

    ~MetalGEMM() { [commandQueue release]; }

    virtual void run()
    {
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];

        [kernel encodeToCommandBuffer:commandBuffer
                           leftMatrix:matA
                          rightMatrix:matB
                         resultMatrix:matC];

        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        [commandBuffer release];
    }

    virtual void init_matrices()
    {
        float *a = static_cast<float *>([bufferA contents]);
        float *b = static_cast<float *>([bufferB contents]);
        float *c = static_cast<float *>([bufferC contents]);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                a[i * n + j] = static_cast<T>(1.0);
                b[i * n + j] = static_cast<T>(1.0);
                c[i * n + j] = static_cast<T>(0.0);
            }
        }
    }
};
