# Skills

This folder contains installable Codex skill bundles that ship with the
Impression documentation.

## Included Skills

- [impression](impression/)
  - use Impression correctly as a modeling framework
  - follow the project's surface-first posture
  - apply the loft rules consistently

## Install

Copy the desired skill folder into your Codex skills directory.

Typical local destination:

```bash
mkdir -p ~/.codex/skills
cp -R docs/skills/impression ~/.codex/skills/
```

After that, agents can invoke:

- `$impression`

The workspace docs remain the source of truth when they are present.
