"""Composition-only ownership/rollback. No numerical operations or task changes."""
from dataclasses import dataclass
import json


@dataclass
class Binding:
    owner: object
    attribute: str
    previous: object
    installed: object
    component: str

    def intact(self):
        return getattr(self.owner,self.attribute) is self.installed

    def restore(self):
        if not self.intact(): return "foreign_preserved"
        setattr(self.owner,self.attribute,self.previous)
        return "restored"


class Ownership:
    """Restore only exact objects installed by this transaction; never foreign hooks."""
    def __init__(self):
        self.bindings=[]

    def add(self, owner, attribute, previous, installed, component):
        if previous is installed: raise RuntimeError(f"{component} did not install {attribute}")
        if any(b.owner is owner and b.attribute==attribute for b in self.bindings):
            raise RuntimeError("Overlapping patch ownership")
        self.bindings.append(Binding(owner,attribute,previous,installed,component))

    def verify(self):
        foreign=[b.component+":"+b.attribute for b in self.bindings if not b.intact()]
        if foreign: raise RuntimeError("Composite patch ownership changed: "+", ".join(foreign))

    def restore(self):
        return {b.component+":"+b.attribute:b.restore() for b in reversed(self.bindings)}


def note(event, **details):
    print("SOUP_FUSION_A "+json.dumps({"event":event,**details},sort_keys=True),flush=True)
