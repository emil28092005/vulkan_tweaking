#!/usr/bin/env bash

set -ex

glslc shaders/shader.vert -o shaders/vert.spv
glslc shaders/shader.frag -o shaders/frag.spv
glslangValidator -V FilmicAnamorphSharpen.fx -o FilmicAnamorphSharpen.spv