"""
Sample extraction from commit data: turns each source file into labelled code
blocks (vulnerable / clean) around the lines changed by a security fix.

Adapted from VUDENC (Wartschinski et al., 2022):
https://github.com/LauraWartschinski/VulnerabilityDetection
"""

SPLITCHARS = [" ", "\t", "\n", ".", ":", "(", ")", "[", "]", "<", ">", "+", "-", "=", "\"", "'",
              "*", "/", "\\", "~", "{", "}", "!", "?", ";", ",", "%", "&"]


def findposition(badpart, sourcecode):
    """Locate a removed ('bad') code fragment in the source, tolerant to whitespace differences."""
    pos = 0
    matchindex = 0
    inacomment = False
    startfound = -1
    endfound = -1
    last = 0

    if "#" in badpart:
        badpart = badpart[:badpart.find("#")]
    if len(badpart.lstrip()) < 1:
        return [-1, -1]

    while True:
        if not inacomment:
            last = pos - 1
        if pos >= len(sourcecode):
            break

        if sourcecode[pos] == "\n":
            inacomment = False
        if sourcecode[pos] == "\n" and (sourcecode[pos - 1] == "\n" or sourcecode[last] == " "):
            pos += 1
            continue
        if sourcecode[pos] == " " and (sourcecode[pos - 1] == " " or sourcecode[last] == "\n"):
            pos += 1
            continue
        if sourcecode[pos] == "#":
            inacomment = True

        if not inacomment:
            a = sourcecode[pos]
            if a == "\n":
                a = " "
            b = badpart[matchindex]
            c = badpart[matchindex - 1] if matchindex > 0 else ""
            d = badpart[matchindex + 1] if matchindex < len(badpart) - 2 else ""

            if a != b and (a == " " or a == "\n") and (b in SPLITCHARS or c in SPLITCHARS):
                pos += 1
                continue
            if a != b and (b == " " or b == "\n"):
                if c in SPLITCHARS or d in SPLITCHARS:
                    if matchindex < len(badpart) - 1:
                        matchindex += 1
                        continue

            if a == b:
                if matchindex == 0:
                    startfound = pos
                matchindex += 1
            else:
                matchindex = 0
                startfound = -1

            if matchindex == len(badpart):
                endfound = pos
                break
        pos += 1

    if endfound < 0:
        return [-1, -1]
    return [startfound, endfound]


def findpositions(badparts, sourcecode):
    positions = []
    for bad in badparts:
        if "#" in bad:
            bad = bad[:bad.find("#")]
        place = findposition(bad, sourcecode)
        if place != [-1, -1]:
            positions.append(place)
    return positions


def nextsplit(sourcecode, focus):
    for pos in range(focus + 1, len(sourcecode)):
        if sourcecode[pos] in SPLITCHARS:
            return pos
    return -1


def previoussplit(sourcecode, focus):
    pos = focus - 1
    while pos >= 0:
        if sourcecode[pos] in SPLITCHARS:
            return pos
        pos -= 1
    return -1


def getcontextPos(sourcecode, focus, fulllength):
    """Grow a window around `focus` (alternating left/right at token boundaries) to > fulllength chars."""
    startcontext = focus
    endcontext = focus
    if focus > len(sourcecode) - 1:
        return None
    start = True
    while not len(sourcecode[startcontext:endcontext]) > fulllength:
        prev, nxt = previoussplit(sourcecode, startcontext), nextsplit(sourcecode, endcontext)
        if prev == -1 and nxt == -1:
            return None
        if start:
            if prev > -1:
                startcontext = prev
            start = False
        else:
            if nxt > -1:
                endcontext = nxt
            start = True
    return [startcontext, endcontext]


def getblocks(sourcecode, badpositions, step, fulllength):
    """
    Slide over the file and return [code_block, label] pairs.
    Label 1 = vulnerable (block overlaps a bad position), 0 = clean.
    """
    blocks = []
    seen = set()
    focus = 0
    lastfocus = 0
    while True:
        if focus > len(sourcecode):
            break

        if sourcecode[lastfocus:focus] != "\n":
            middle = lastfocus + round(0.5 * (focus - lastfocus))
            context = getcontextPos(sourcecode, middle, fulllength)
            if context is not None:
                vulnerable = any(
                    (bad[0] < context[0] <= bad[1]) or (bad[0] < context[1] <= bad[1])
                    or (context[0] <= bad[0] and context[1] >= bad[1])
                    for bad in badpositions
                )
                code = sourcecode[context[0]:context[1]]
                if code not in seen:
                    seen.add(code)
                    blocks.append([code, 1 if vulnerable else 0])

        if "\n" in sourcecode[focus + 1:focus + 7]:
            lastfocus = focus
            focus = focus + sourcecode[focus + 1:focus + 7].find("\n") + 1
        else:
            nxt = nextsplit(sourcecode, focus + step)
            if nxt > -1:
                lastfocus = focus
                focus = nxt
            elif focus < len(sourcecode):
                lastfocus = focus
                focus = len(sourcecode)
            else:
                break
    return blocks
