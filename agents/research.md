# Research Guidance

Research documents exist so agents can store reference information locally when they have been instructed to do research.

Agents are unlikely to independently choose to do research, and research that is not saved into the project will fall out of working context.

The purpose of the research folder is to make research durable and project-local so it can be used in the execution of the project.

---

## Location
project/research

## Purpose

Research documents preserve information that has been gathered for the project but is not yet part of architecture, specifications, planning, or code.

They exist so that:

* researched information remains available after the current context is gone
* future work can build on prior research
* agents can use project-local reference material instead of repeatedly rediscovering the same information

---

## Scope

Research may include:

* library or tool behavior
* external system constraints
* experimental findings
* implementation notes
* distilled reference material

Research should contain information that may be useful to later architectural, specification, planning, or implementation work.

---

## Organization
Use folder structure to organize documents. Use the top level directory to relate to specific project topics.

## Use

Research is durable working information used in the execution of the project.

It supports:

* architectural refinement
* specification writing
* planning
* implementation

Research is not itself the architectural or implementation definition of the system.

If research becomes part of the defined system, it should be referenced in the corresponding architecture or specification documents.

---

## Recommended Structure

Research documents should generally include:

### Topic

What was researched.

### Findings

What was learned.

### Implications

How it may affect the project.

### References

Source material or origin of the findings.

---

## Relationship to Other Documents

* architecture defines system structure
* specifications define implementation
* planning defines sequencing and assignment
* research stores durable reference information used by all of them

---

## Guiding Principle

Research is durable project knowledge created so agents can retain and reuse information gathered for the project.
