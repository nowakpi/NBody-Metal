//
//  shaders.metal
//  NBody-Metal
//
//  Created by James Price on 09/10/2015.
//  Copyright © 2015 James Price. All rights reserved.
//
//  Modified by Piotr Nowak on 01/05/2019.
//  Copyright © 2019 Piotr Nowak. All rights reserved.
//


#include <metal_stdlib>
using namespace metal;

#define GROUPSIZE 64 // must be same as GROUPSIZE in NBodyViewController.swift

struct Params
{
  uint  nbodies;
  float delta;
  float softening;
};

float4 computeForce(float4 ipos, float4 jpos, float softening);

float4 computeForce(float4 ipos, float4 jpos, float softening)
{
  float4 d      = jpos - ipos;
         d.w    = 0;
  float  distSq = d.x*d.x + d.y*d.y + d.z*d.z + softening*softening;
  float  dist   = fast::rsqrt(distSq);
  float  coeff  = jpos.w * (dist*dist*dist);
  return coeff * d;
}

float4 computeColor(float4 velocityVector, float defaultIntensity);

float4 computeColor(float4 velocityVector, float defaultIntensity)
{
    float4 r = float4(defaultIntensity, defaultIntensity, defaultIntensity, defaultIntensity);
    float  distSq = velocityVector.x*velocityVector.x + velocityVector.y*velocityVector.y + velocityVector.z*velocityVector.z;
    float  dist   = fast::rsqrt(distSq);
    
    if (dist <= 0.0015) {
        //white for the very fast
        r.x = 0.88;
        r.y = 0.88;
        r.z = 0.88;
    } else if (dist <= 0.0045) {
        r.x = 0.15 + (dist / 0.0045) * 0.80;
        r.y = 0.10 + (dist / 0.0045) * 0.15;
        r.z = 0.10 + (dist / 0.0045) * 0.25;
    } else if (dist > 0.01) {
        r.x = 0.10 + (0.01 / dist) * 0.15;
        r.y = 0.01 + (0.01 / dist) * 0.10;
        r.z = 0.10 + (0.01 / dist) * 0.70;
    } else {
        float scDist = dist * 10000;
        r.x = 0.05 + (scDist / 55) * 0.15;
        r.y = 0.05 + (scDist / 55) * 0.55;
        r.z = 0.75 - (scDist / 55) * 0.45;
    }
    
    return r;
}

struct VertexIn {
    float4 position [[attribute(0)]];
    float4 color [[attribute(1)]];
};

struct VertexInJustForMemoryLayout {
    float4 position;
    float4 color;
};

kernel void step(const device VertexIn* positionsIn  [[buffer(0)]],
                       device VertexInJustForMemoryLayout* positionsOut [[buffer(1)]],
                       device   float4* velocities   [[buffer(2)]],
                       constant Params  &params      [[buffer(3)]],
                                uint    i            [[thread_position_in_grid]],
                                uint    l            [[thread_position_in_threadgroup]])
{
  float4 ipos = positionsIn[i].position;

  threadgroup float4 scratch[GROUPSIZE];

  // Compute force
  float4 force = 0.f;
  for (uint j = 0; j < params.nbodies; j+=GROUPSIZE)
  {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    scratch[l] = positionsIn[j + l].position;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 0; k < GROUPSIZE;)
    {
      force += computeForce(ipos, scratch[k++], params.softening);
      force += computeForce(ipos, scratch[k++], params.softening);
      force += computeForce(ipos, scratch[k++], params.softening);
      force += computeForce(ipos, scratch[k++], params.softening);
    }
  }

  // Update velocity
  float4 velocity = velocities[i];
  velocity       += force * params.delta;
  velocities[i]   = velocity;

  // Update position
    positionsOut[i].position = ipos + velocity*params.delta;
    positionsOut[i].color = computeColor(velocities[i], 0.6);
}

#define POINT_SCALE 5.f
#define SIGHT_RANGE  2.f

struct VertexOut
{
  float4 position  [[position]];
  float  pointSize [[point_size]];
  float4 color;
};

struct RenderParams
{
  float4x4 vpMatrix;
  float3   eyePosition;
};

vertex VertexOut vert(const VertexIn vertexIn [[stage_in]], const device RenderParams &params  [[buffer(1)]] )
{
  VertexOut out;

  float4 pos = vertexIn.position;
  out.position = params.vpMatrix * pos;

    float dist = distance(pos.xyz, params.eyePosition);
    float size = POINT_SCALE * (1.f - (dist / SIGHT_RANGE));
    out.pointSize = max(size, 0.f);

    
  out.color = vertexIn.color;

  return out;
}

fragment half4 frag(VertexOut vertexIn [[stage_in]], float2 pointCoord [[point_coord]])
{
    return half4(vertexIn.color.x, vertexIn.color.y, vertexIn.color.z, vertexIn.color.w);
}
