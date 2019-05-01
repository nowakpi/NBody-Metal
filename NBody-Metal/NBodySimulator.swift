//
//  NBodySimulator.swift
//  NBody-Metal
//
//  Created by Piotr Nowak on 01/05/2019.
//


import Foundation
import AppKit
import Metal
import MetalKit


//Global scope functions/operators/structs
func _s(_ t : String, _ np : String = "") -> String {
    let n = "0123456789" + np //["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    var r = ""
    for c in t {
        if n.contains(c) { r.append(c) }
    }
    return r
}

func getTimestamp() -> Double {
    var tv:timeval = timeval()
    gettimeofday(&tv, nil)
    return (Double(tv.tv_sec)*1e3 + Double(tv.tv_usec)*1e-3)
}

infix operator +/-
func +/-(lhs: Float, rhs: Float) -> Float {
    return getTimestamp().truncatingRemainder(dividingBy: 2.0) == 0 ? lhs+rhs : lhs-rhs
}

struct Vertex {
    var pos: float4
    var col: float4
}

protocol VDataRefresher {
    func setDeviceName(_ as : String)
    func setNbodiesText(_ as : String)
    func attachViewTo(_ dev : MTLDevice)
    func getSofteningText() -> String?
}

//The Simulator code
final class NBodySimulator {
    
    public  var viewDelegate : VDataRefresher?
    
    public  let WIDTH     = 1600
    public  let HEIGHT    = 900
    private let RADIUS    = Float(0.5)
    private let GROUPSIZE = 64 // must be same as GROUPSIZE in shaders.metal
    private let DELTA     = Float(0.00001)
    private let SOFTENING = Float(0.005)
    private let MAXBODIES = 320768
    private let MINBODIES = 64 // must be same as GROUPSIZE in shaders.metal
    
    private var deviceIndex : Int = 0
    public  var nbodies     = 72192
    public  var power : Bool = false
    private var queue: MTLCommandQueue?
    private var library: MTLLibrary!
    private var computePipelineState: MTLComputePipelineState!
    private var renderPipelineState: MTLRenderPipelineState!
    private var buffer: MTLCommandBuffer?
    
    private var d_positions0: MTLBuffer?
    private var d_positions1: MTLBuffer?
    private var d_velocities: MTLBuffer?
    
    private var d_positionsIn:  MTLBuffer?
    private var d_positionsOut: MTLBuffer?
    
    private var d_computeParams: MTLBuffer!
    private var d_renderParams:  MTLBuffer!
    
    private var projectionMatrix: Matrix4!
    
    final private func ensureValue(_ seed: Int) -> Float {
        let ing : Int = seed > 0 ? seed : MAXBODIES
        let lhs = Float(1 - 1/ing)
        let rhs = Float(arc4random())/Float(RAND_MAX)
        let ret = lhs +/- rhs
        return ret != 0.0 ? ret : rhs
    }
    
    final private func ev(_ v : Float?) -> Float {
        var r = Float(0.0)
        if let vu = v, !vu.isNaN { r = vu }
        return r
    }
    
    final private func norm(_ f : Float) -> Float {
        let maxRadius : Float = 2.0 //so values would be from -3 to 3, so lenght 2xradius
        return f / maxRadius
    }
    
    final private func reb_random_uniform(min : Double, max : Double) -> Double {
        return Double(arc4random()) / Double(RAND_MAX)*(max-min) + min
    }
    
    final private func reb_random_powerlaw(min : Double, max : Double, slope : Double) -> Double {
        let y : Double = reb_random_uniform(min: 0.0, max: 1.0)
        
        if (slope == -1) {
            return exp(y*log(max/min) + log(min))
        }
        else {
            return pow( (pow(max,slope+1.0)-pow(min,slope+1.0))*y+pow(min,slope+1.0), 1.0/(slope+1.0))
        }
    }
    
    final private func reb_random_normal(variance : Double) -> Double {
        var v1 : Double = 1.0, v2 : Double = 1.0, rsq : Double = 1.0
        
        while (rsq >= 1.0 || rsq < 1.0e-12) {
            v1 = (2.0 * Double(arc4random()) / Double(RAND_MAX) - 1.0)
            v2 = (2.0 * Double(arc4random()) / Double(RAND_MAX) - 1.0)
            rsq = v1*v1 + v2*v2;
        }
        // Note: This gives another random variable for free, but we'll throw it away for simplicity and for thread-safety.
        return v1 * sqrt(-2.0*log(rsq)/rsq*variance)
    }
    
    final public func computeRender(on view: MTKView) {
        buffer = queue?.makeCommandBuffer()
        
        // Compute kernel
        let groupsize = MTLSizeMake(GROUPSIZE, 1, 1)
        let numgroups = MTLSizeMake(nbodies/GROUPSIZE, 1, 1)
        
        if let computeEncoder = buffer!.makeComputeCommandEncoder() {
            
            computeEncoder.setComputePipelineState(computePipelineState)
            computeEncoder.setBuffer(d_positionsIn, offset: 0, index: 0)
            computeEncoder.setBuffer(d_positionsOut, offset: 0, index: 1)
            computeEncoder.setBuffer(d_velocities, offset: 0, index: 2)
            computeEncoder.setBuffer(d_computeParams, offset: 0, index: 3)
            
            computeEncoder.dispatchThreadgroups(numgroups, threadsPerThreadgroup: groupsize)
            computeEncoder.endEncoding()
            
            // Vertex and fragment shaders
            let renderPassDescriptor = view.currentRenderPassDescriptor
            renderPassDescriptor!.colorAttachments[0].loadAction = .clear
            renderPassDescriptor!.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.1, 1.0)
            
            
            if let renderEncoder  = buffer!.makeRenderCommandEncoder(descriptor: renderPassDescriptor!) {
                
                renderEncoder.setRenderPipelineState(renderPipelineState)
                renderEncoder.setVertexBuffer(d_positionsIn, offset: 0, index: 0)
                renderEncoder.setVertexBuffer(d_renderParams, offset: 0, index: 1)
                
                //renderEncoder.setFragmentBuffer(d_velocities, offset: 0, index: 0)
                
                //            let h_positions  = d_positionsOut!.contents().assumingMemoryBound(to: Float.self)
                //            let h_velocities = d_velocities!.contents().assumingMemoryBound(to: Float.self)
                //
                //            print("drawMTKView draw:")
                //            print(h_positions)
                
                //            var bodies = [(p1: Float,p2: Float, p3: Float, p4: Float, c1: Float, c2: Float, c3: Float, c4: Float, v1: Float, v2: Float, v3: Float, v4: Float)]()
                //
                //            for i in 0...100 {
                //                if h_velocities[i*4 + 1] > 0.0 {
                //                    bodies.append((p1: h_positions[i*8 + 0], p2: h_positions[i*8 + 1], p3: h_positions[i*8 + 2], p4: h_positions[i*8 + 3],
                //                                   c1: h_positions[i*8 + 4], c2: h_positions[i*8 + 5], c3: h_positions[i*8 + 6], c4: h_positions[i*8 + 7],
                //                                   v1: h_velocities[i*4 + 0], v2: h_velocities[i*4 + 1], v3: h_velocities[i*4 + 2], v4: h_velocities[i*4 + 3] ))
                //                }
                //            }
                //
                //            for body in bodies {
                //                print(body)
                //            }
                
                renderEncoder.setBlendColor(red: 0.4, green: 0.4, blue: 0.5, alpha: 0.0)
                renderEncoder.drawPrimitives(type: .point, vertexStart: 0, vertexCount: nbodies)
                
                renderEncoder.endEncoding()
            }
        }
        
        buffer!.present(view.currentDrawable!)
        buffer!.commit()
        
        swap(&d_positionsIn, &d_positionsOut)
    }
    
    final public func initBodies(distX : String = "RL", distY : String = "RL", distZ : String = "RL", veloX : String = "RL", veloY : String = "RL", veloZ : String = "RL", discMass : Double = 11e-8, boxSize : Double = 2.4, velocityStrenght : Float = 64.0) {
        
        buffer?.waitUntilCompleted()
        
        // Initialise positions uniformly at random on surface of sphere, with no velocity
        let h_positions  = d_positionsIn!.contents().assumingMemoryBound(to: Float.self)
        let h_velocities = d_velocities!.contents().assumingMemoryBound(to: Float.self)
        
        print("initBodies:")
        print(h_positions)
        
        
        var minX : Float = 0.0
        var maxX : Float = 0.0
        var minY : Float = 0.0
        var maxY : Float = 0.0
        
        for i in 0...(nbodies-1) {
            
            let longitude = 2.0 * Float(Double.pi) * (Float(arc4random())/Float(RAND_MAX))
            let latitude  = acos((2.0 * (Float(arc4random())/Float(RAND_MAX))) - 1.0)
            
            let aV : Double = reb_random_powerlaw(min: boxSize/10.0, max: boxSize/2.0/1.2, slope: -1.5)
            let phi : Double = reb_random_uniform(min: 0.0, max: 2.0*Double.pi)
            
            switch distX {
            case "RSC": h_positions[i*8 + 0] = RADIUS * sin(latitude) * cos(longitude)
            case "RSS": h_positions[i*8 + 0] = RADIUS * sin(latitude) * sin(longitude)
            case "RC": h_positions[i*8 + 0] = RADIUS * cos(latitude)
            case "RL": h_positions[i*8 + 0] = ensureValue(i)
            case "R0":  h_positions[i*8 + 0] = 0.0
            case "CRR": fallthrough
            default: h_positions[i*8 + 0] = (Float(aV * cos(phi)))
            }
            
            switch distY {
            case "RSC": h_positions[i*8 + 1] = RADIUS * sin(latitude) * cos(longitude)
            case "RSS": h_positions[i*8 + 1] = RADIUS * sin(latitude) * sin(longitude)
            case "RC": h_positions[i*8 + 1] = RADIUS * cos(latitude)
            case "RL": h_positions[i*8 + 1] = ensureValue(i)
            case "R0": h_positions[i*8 + 1] = 0.0
            case "CRR": fallthrough
            default: h_positions[i*8 + 1] = (Float(aV * sin(phi)))
            }
            
            switch distZ {
            case "RSC": h_positions[i*8 + 2] = RADIUS * sin(latitude) * cos(longitude)
            case "RSS": h_positions[i*8 + 2] = RADIUS * sin(latitude) * sin(longitude)
            case "RC": h_positions[i*8 + 2] = RADIUS * cos(latitude)
            case "RL": h_positions[i*8 + 2] = ensureValue(i)
            case "R0": h_positions[i*8 + 2] = 0.0
            case "CRR": h_positions[i*8 + 2] = 0.0
            default: h_positions[i*8 + 2] = 0.0
            }
            
            h_positions[i*8 + 3] = 1.0
            
            h_positions[i*8 + 4] = 0.18 // R ?
            h_positions[i*8 + 5] = 0.19 //G ?
            h_positions[i*8 + 6] = 0.75 //B?
            h_positions[i*8 + 7] = 1.0 //A ?
            
            //noise feature is a cover for mathematical bug
            if h_positions[i*8 + 0].isNaN || (distX != "R0" && h_positions[i*8 + 0].isZero) {
                h_positions[i*8 + 0] = ensureValue(i)
            }
            if h_positions[i*8 + 1].isNaN || (distY != "R0" && h_positions[i*8 + 1].isZero) {
                h_positions[i*8 + 1] = ensureValue(i)
            }
            if h_positions[i*8 + 2].isNaN || (distZ != "R0" && h_positions[i*8 + 2].isZero) {
                h_positions[i*8 + 2] = ensureValue(i)
            }
            
            let muVi : Double = discMass * (pow(aV, -3.0/2.0)-pow(boxSize/10.0, -3.0/2.0)) / (pow(boxSize/2.0/1.2, -3.0/2.0) - pow(boxSize/10.0, -3.0/2.0))
            let vKep : Double = sqrt(muVi/aV)
            
            switch veloX {
            case "RSC": h_velocities[i*4 + 0] = -100 * ev(sin(latitude)) * Float(arc4random())/Float(RAND_MAX)  + 100*ev(cos(longitude)) * Float(arc4random())/Float(RAND_MAX)
            case "RSS": h_velocities[i*4 + 0] = (-25.0 +/- -ev(sin(latitude))) + ev(sin(latitude)) * 100.0
            case "R0": h_velocities[i*4 + 0] = 0.0
            case "RL": h_velocities[i*4 + 0] = 1 * (1.0 +/- 100.0) * Float(arc4random())/Float(RAND_MAX)
            case "CR":
                h_velocities[i*4 + 0] = 0.0 //let's not leave what was there before
                if h_positions[i*8 + 1] >= 0.0 { //when x position is positive
                    h_velocities[i*4 + 0] = sin(Float.pi/2 + norm(h_positions[i*8 + 0]))*velocityStrenght
                } else {
                    h_velocities[i*4 + 0] = -sin(Float.pi/2 - norm(h_positions[i*8 + 0]))*velocityStrenght
                }
            case "CRR":
                h_velocities[i*4 + 0] = Float(vKep * sin(phi) * 5625.0 * Double(velocityStrenght))
            default: h_velocities[i*4 + 0] = 100
            }
            
            switch veloY {
            case "RSC": h_velocities[i*4 + 1] = -100 * ev(sin(latitude)) * Float(arc4random())/Float(RAND_MAX)  + 100*ev(cos(longitude)) * Float(arc4random())/Float(RAND_MAX)
            case "RSS": h_velocities[i*4 + 1] = (-55.0 +/- ev(sin(latitude))) + ev(sin(latitude)) * 100.0
            case "R0": h_velocities[i*4 + 1] = 0.0
            case "RL": h_velocities[i*4 + 1] = 1 * (60.0 +/- 100.0) * Float(arc4random())/Float(RAND_MAX)
            case "CR":
                h_velocities[i*4 + 1] = 0.0 //let's not leave what was there before
                if h_positions[i*8 + 0] >= 0.0 { //when x position is positive
                    h_velocities[i*4 + 1] = -cos(Float.pi/2 - norm(h_positions[i*8 + 1]))*velocityStrenght
                } else {
                    h_velocities[i*4 + 1] = cos(Float.pi/2 + norm(h_positions[i*8 + 1]))*velocityStrenght
                }
            case "CRR":
                h_velocities[i*4 + 1] = Float(-vKep * cos(phi)) * 5625.0 * velocityStrenght
            default: h_velocities[i*4 + 1] = 10
            }
            
            switch veloZ {
            case "RSC": h_velocities[i*4 + 2] = -100 * ev(sin(latitude)) * Float(arc4random())/Float(RAND_MAX)  + 100*ev(cos(longitude)) * Float(arc4random())/Float(RAND_MAX)
            case "RSS": h_velocities[i*4 + 2] = (-20.0 +/- ev(sin(latitude))) + ev(cos(latitude)) * 100.0
            case "R0": h_velocities[i*4 + 2] = 0.0
            case "RL": h_velocities[i*4 + 2] = 1 * (1.0 +/- 100.0) * Float(arc4random())/Float(RAND_MAX)
            case "CR": h_velocities[i*4 + 2] = 0.0
            case "CRR": h_velocities[i*4 + 2] = 0.0 //in CR and CRR Z needs to be zero as 3D transformation like that is not supported yet ;)
            default: h_velocities[i*4 + 2] = 10
            }
            
            //        pt.x         = a*cos(phi);
            //        pt.y         = a*sin(phi);
            //        pt.z         = a*reb_random_normal(0.001);
            
            h_velocities[i*4 + 3] = Float(discMass/(Double(nbodies)/2)) //mass
            
            //h_velocities[i*4 + 0] = 60
            //h_velocities[i*4 + 1] = 60
            //h_velocities[i*4 + 2] = 0.0
            //h_velocities[i*4 + 3] = 20.0
            
            if h_positions[i*8 + 0] > maxX { maxX = h_positions[i*8 + 0]}
            if h_positions[i*8 + 0] < minX { minX = h_positions[i*8 + 0]}
            if h_positions[i*8 + 1] > maxY { maxY = h_positions[i*8 + 1]}
            if h_positions[i*8 + 1] < minY { minY = h_positions[i*8 + 1]}
        }
        
        print(minX, maxX, minY, maxY)
        
        d_positionsIn?.didModifyRange(0..<MemoryLayout<Vertex>.stride * nbodies)
        d_velocities?.didModifyRange(0..<MemoryLayout<Vertex>.stride * nbodies) //sizeof(float4)
    }
    
    final public func initMetal(retainBodies: Bool) {
        
        // Get data from previous device
        let h_positions  = d_positionsIn?.contents()
        let h_velocities = d_velocities?.contents()
        let buffer       = queue?.makeCommandBuffer()
        let blitEncoder  = buffer?.makeBlitCommandEncoder()
        blitEncoder?.synchronize(resource: d_positionsIn!)
        blitEncoder?.synchronize(resource: d_velocities!)
        blitEncoder?.endEncoding()
        buffer?.commit()
        buffer?.waitUntilCompleted()
        
        // Select next device
        let device = MTLCopyAllDevices()[deviceIndex]
        
        viewDelegate?.setDeviceName("Device: \(device.name) [d]")
        viewDelegate?.setNbodiesText("Bodies: \(nbodies) [+/-]")
        viewDelegate?.attachViewTo(device)
        
        print(device.isHeadless)
        print(device.isLowPower)
        print(device.isRemovable)
        print(device.recommendedMaxWorkingSetSize)
        print(device.currentAllocatedSize)
        print(device.maxBufferLength)
        
        queue      = device.makeCommandQueue()
        library    = device.makeDefaultLibrary()
        do {
            computePipelineState = try device.makeComputePipelineState(function: library.makeFunction(name: "step")!)
        }
        catch {
            print("Failed to create compute pipeline state")
        }
        
        let renderPipelineStateDescriptor = MTLRenderPipelineDescriptor()
        renderPipelineStateDescriptor.vertexFunction = library.makeFunction(name: "vert")
        renderPipelineStateDescriptor.fragmentFunction = library.makeFunction(name: "frag")
        renderPipelineStateDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        renderPipelineStateDescriptor.colorAttachments[0].isBlendingEnabled = false
        renderPipelineStateDescriptor.colorAttachments[0].destinationRGBBlendFactor = .one
        renderPipelineStateDescriptor.colorAttachments[0].destinationAlphaBlendFactor = .one
        
        let vertexDescriptor = MTLVertexDescriptor()
        vertexDescriptor.attributes[0].format = .float4
        vertexDescriptor.attributes[0].offset = 0
        vertexDescriptor.attributes[0].bufferIndex = 0
        vertexDescriptor.attributes[1].format = .float4
        vertexDescriptor.attributes[1].offset = MemoryLayout<float4>.stride
        vertexDescriptor.attributes[1].bufferIndex = 0
        vertexDescriptor.layouts[0].stride = MemoryLayout<Vertex>.stride
        
        renderPipelineStateDescriptor.vertexDescriptor = vertexDescriptor
        
        do {
            renderPipelineState = try device.makeRenderPipelineState(descriptor: renderPipelineStateDescriptor)
        }
        catch {
            print("Failed to create render pipeline state")
        }
        
        // Create device buffers
        let datasize = MemoryLayout<Vertex>.stride * nbodies //sizeof(float4)*nbodies
        d_positions0 = device.makeBuffer(length: datasize, options: .storageModeManaged)
        d_positions1 = device.makeBuffer(length: datasize, options: .storageModeManaged)
        d_velocities = device.makeBuffer(length: datasize, options: .storageModeManaged)
        
        // Copy data from previous device
        if retainBodies {
            if h_positions != nil {
                memcpy(d_positions0!.contents(), h_positions!, datasize)
                d_positions0?.didModifyRange(0..<datasize)
            }
            if h_velocities != nil {
                memcpy(d_velocities!.contents(), h_velocities!, datasize)
                d_velocities?.didModifyRange(0..<datasize)
            }
        }
        
        d_positionsIn  = d_positions0
        d_positionsOut = d_positions1
        
        struct ComputeParams {
            var nbodies:UInt32  = 0
            var delta:Float     = 0
            var softening:Float = 0
        }
        
        var soft = SOFTENING
        if let s=viewDelegate?.getSofteningText(), let z=Float(_s(s,".")) { soft = z }
        
        var h_computeParams = ComputeParams(nbodies: UInt32(nbodies), delta: DELTA, softening: soft)
        d_computeParams = device.makeBuffer(bytes: &h_computeParams, length: MemoryLayout<ComputeParams>.size, options: [])
        
        // Initialise view-projection matrices
        guard let vpMatrix = Matrix4() else { return }
        
        vpMatrix.translate(0.0, y: 0.0, z: -2.0)
        projectionMatrix = Matrix4.makePerspectiveViewAngle(Matrix4.degrees(toRad: 55.0), aspectRatio: Float(WIDTH)/Float(HEIGHT), nearZ: 0.1, farZ: 50.0)
        vpMatrix.multiplyLeft(projectionMatrix)
        
        var eyePosition = float3(0, 0, 2.0)
        
        let renderParamsSize = MemoryLayout<matrix_float4x4>.size + MemoryLayout<Float>.size * 4
        
        d_renderParams = device.makeBuffer(length: renderParamsSize, options: [])
        memcpy(d_renderParams.contents(), vpMatrix.raw(), MemoryLayout<matrix_float4x4>.size)
        memcpy(d_renderParams.contents() + MemoryLayout<matrix_float4x4>.size, &eyePosition, MemoryLayout<float3>.size)
    }

    final public func selectNextDevice() {
        // Select next device
        deviceIndex = deviceIndex + 1
        if deviceIndex >= MTLCopyAllDevices().count {
            deviceIndex = 0
        }
        initMetal(retainBodies: true)
    }
    
    final public func lessBodies() {
        if nbodies > MINBODIES {
            nbodies /= 2
            initMetal(retainBodies: false)
            initBodies()
        }
    }
    
    final public func moreBodies() {
        if nbodies < MAXBODIES {
            nbodies *= 2
            initMetal(retainBodies: false)
            initBodies()
        }
    }
}
