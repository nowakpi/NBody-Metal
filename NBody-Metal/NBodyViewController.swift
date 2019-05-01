//
//  NBodyViewController.swift
//  NBody-Metal
//
//  Created by James Price on 09/10/2015.
//  Copyright © 2015 James Price. All rights reserved.
//
//  Modified by Piotr Nowak on 01/05/2019.
//  Copyright © 2019 Piotr Nowak. All rights reserved.
//


import Foundation
import AppKit
import Metal
import MetalKit

extension NSView {
    func retrieveSubviewBy(name : String) -> NSView? {
        var r : NSView?; for v in self.subviews { if v.identifier?.rawValue == name {r = v} }
        return r
    }
}

final class NBodyViewController: NSViewController, MTKViewDelegate, VDataRefresher {
    
    @IBOutlet public var powerBt : NSButton?
    @IBOutlet public var regenBt : NSButton?
    @IBOutlet public var distXCmb : NSComboBox?
    @IBOutlet public var distYCmb : NSComboBox?
    @IBOutlet public var distZCmb : NSComboBox?
    @IBOutlet public var veloXCmb : NSComboBox?
    @IBOutlet public var veloYCmb : NSComboBox?
    @IBOutlet public var veloZCmb : NSComboBox?
    @IBOutlet public var simulationView : NSView?
    @IBOutlet public var numBdTxt : NSTextField?
    @IBOutlet public var massTxt : NSTextField?
    @IBOutlet public var boxTxt : NSTextField?
    @IBOutlet public var strengthTxt : NSTextField?
    @IBOutlet public var softeningTxt : NSTextField?
    
    private var nametext : NSTextField!
    private var nbodiestext : NSTextField!
    private var fpstext : NSTextField!
    private var flopstext : NSTextField!
    
    private var metalview : MTKView!
    private var simulator : NBodySimulator!
    
    private var frames = 0
    private var lastUpdate : Double = 0
    
    override func viewDidDisappear() {
        super.viewDidDisappear()
        do { NSApplication.shared.terminate(nil) }
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        // Add view controller to responder chain
        self.view.window?.nextResponder = self
        self.nextResponder = nil
        
        // Create MTKView object based on Simulator
        simulator = NBodySimulator()
        simulator.viewDelegate = self
        
        metalview = MTKView(frame: CGRect(x: 0, y: 0, width: simulator.WIDTH, height: simulator.HEIGHT))
        metalview.delegate = self
        
        simulationView!.addSubview(metalview)
        
        // Create status labels
        nametext    = createInfoText(rect: NSMakeRect(10, CGFloat(simulator.HEIGHT)-30, 300, 20))
        nbodiestext = createInfoText(rect: NSMakeRect(10, CGFloat(simulator.HEIGHT)-50, 300, 20))
        fpstext     = createInfoText(rect: NSMakeRect(10, CGFloat(simulator.HEIGHT)-70, 120, 20))
        flopstext   = createInfoText(rect: NSMakeRect(10, CGFloat(simulator.HEIGHT)-90, 120, 20))
        
        metalview.addSubview(nametext)
        metalview.addSubview(nbodiestext)
        metalview.addSubview(fpstext)
        metalview.addSubview(flopstext)
        
        distXCmb?.selectItem(at: 0)
        distYCmb?.selectItem(at: 0)
        distZCmb?.selectItem(at: 0)
        
        veloXCmb?.selectItem(at: 0)
        veloYCmb?.selectItem(at: 0)
        veloZCmb?.selectItem(at: 0)
        
        simulator.initMetal(retainBodies: false)
        simulator.initBodies()
    }
    
    func createInfoText(rect: NSRect) -> NSTextField {
        let text = NSTextField(frame: rect)
        text.isEditable        = false
        text.isBezeled         = false
        text.isSelectable      = false
        text.drawsBackground = false
        text.textColor       = NSColor.white
        text.font            = NSFont.boldSystemFont(ofSize: 14.0)
        text.stringValue     = ""
        
        return text
    }
    
    func draw(in view: MTKView) {
        guard simulator.power == true else { return }
        self.drawInMTKView(view: view)
    }
    
    @IBAction func regenerateAction(_ sender: NSButton) {
        guard self.simulator.power == false else { return }
        self.powerBt?.isEnabled = false
        
        var mass, box : Double?
        var vel : Float?
        if let s = numBdTxt?.stringValue, let z = Int(_s(s)) { simulator.nbodies = z }
        if let s = massTxt?.stringValue, let z = Double(_s(s, "-e")) { mass = z }
        if let s = boxTxt?.stringValue, let z = Double(_s(s, ".")) { box = z }
        if let s = strengthTxt?.stringValue, let z = Float(_s(s, ".")) { vel = z }
        
        //11e-8, boxSize : Double = 2.4, velocityStrenght : Float = 64.0)
        
        simulator.initMetal(retainBodies: false)
        simulator.initBodies(distX: distXCmb!.stringValue,
                        distY: distYCmb!.stringValue,
                        distZ: distZCmb!.stringValue,
                        veloX: veloXCmb!.stringValue,
                        veloY: veloYCmb!.stringValue,
                        veloZ: veloZCmb!.stringValue,
                        discMass: mass != nil ? mass! : 11e-8,
                        boxSize: box != nil ? box! : 2.4,
                        velocityStrenght: vel != nil ? vel! : 64.0)
        
        simulator.power = true
        let deadlineTime = DispatchWallTime.now() + .milliseconds(30)
        DispatchQueue.global().asyncAfter(wallDeadline: deadlineTime) {
            self.simulator.power = false
            DispatchQueue.main.async { self.powerBt?.isEnabled = true }
        }
    }
    
    @IBAction func powerOnOffAction(_ sender: NSButton) {
        self.simulator.power = !self.simulator.power
        self.regenBt?.isEnabled = !self.simulator.power
    }
    
    func drawInMTKView(view: MTKView) {
        // Update FPS and GFLOP/s counters
        frames += 1
        let now  = getTimestamp()
        let diff = now - lastUpdate
        if diff >= 1000 {
            let fps = (Double(frames) / diff) * 1000
            let strfps = NSString(format: "%.1f", fps)
            fpstext.stringValue = "FPS: \(strfps)"
            
            let flopsPerPair = 21.0
            let gflops = ((Double(frames) * Double(simulator.nbodies) * Double(simulator.nbodies) * flopsPerPair) / diff) * 1000 * 1e-9
            let strflops = NSString(format: "%.1f", gflops)
            flopstext.stringValue = "GFLOP/s: \(strflops)"
            
            frames = 0
            lastUpdate = now
        }
        
        simulator.computeRender(on: view)
    }
    
    override func keyDown(with: NSEvent) {
        switch with.keyCode {
        case 2:   simulator.selectNextDevice()
        case 15:  simulator.initBodies()
        case 12:  exit(0)
        case 27:  simulator.lessBodies()
        case 24:  with.modifierFlags.contains(NSEvent.ModifierFlags.shift) ? simulator.moreBodies() : ()
        default:  super.keyDown(with: with)
        }
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        
    }
    
    final public func setDeviceName(_ asN: String) {
        nametext.stringValue = asN
    }
    
    final public func setNbodiesText(_ asT: String) {
        nbodiestext.stringValue = asT
    }
    
    final public func attachViewTo(_ dev: MTLDevice) {
        metalview.device = dev
    }
    
    final public func getSofteningText() -> String? {
        return softeningTxt?.stringValue
    }
}
