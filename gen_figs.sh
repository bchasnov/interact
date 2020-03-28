#!/bin/sh

composite $1_stable.png -compose Multiply $1_nash.png $1.png
