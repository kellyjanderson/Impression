# Testing Spec 18 Test: Computer Vision Canonical Object-View Set and Authoritative Derived Products

## Overview

This test specification defines verification for the initial canonical
object-view set and authoritative derived product bundle.

## Backlink

- [Testing Spec 18: Computer Vision Canonical Object-View Set and Authoritative Derived Products (v1.0)](../specifications/testing-18-computer-vision-canonical-object-view-set-and-authoritative-derived-products-v1_0.md)

## Automated Smoke Tests

- representative fixtures emit the declared canonical view set in stable order
- authoritative derived products are emitted deterministically
- optional beauty renders remain separable from authoritative products

## Automated Acceptance Tests

- missing or reordered canonical views fail clearly
- authoritative products remain comparable across repeated runs
- beauty renders do not silently become the only proof artifact
- naming and ordering remain stable across fixtures using the same lane
