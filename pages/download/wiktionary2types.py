# Copyright (c) 2014, G. Wisniewski, L. Aufrant
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Extract information from Wikitionary dumps

For the moment only POS information are extracted even if other
information (such as pronounciation and translation are included in
Wikitionary).

The main difficulties for the extraction are:
- Wiktionary dumps include help pages, index and other information
  that are not dictionary entries
- Each entry in a given language describes the word in that language
  (i.e.  its different POS and meanings) but also in other
  languages. For instance the Spanish page for “mes” contains the
  description of the Spanish word (“month”), the French word (“our”)
  as well as the description of “mes” in 14 other languages. All
  information are contains in the same page and separated either by
  wiki templates or by level 2 titles.

Documentation about Wiktionary page organization can be found:
- http://en.wiktionary.org/wiki/Wiktionary:Headword-line_templates
- http://en.wiktionary.org/wiki/Wiktionary:Entry_layout_explained/POS_headers
- http://en.wiktionary.org/wiki/Wiktionary:Entry_layout_explained
- http://en.wiktionary.org/wiki/Help:FAQ
"""
import functools
import logging
import re
import regex 

from functools import partial
from itertools import chain
from collections import defaultdict

from xml.etree.cElementTree import iterparse

import requests
import requests_cache

requests_cache.install_cache()


# indicates a link between two Wiktionary entries (used, e.g. to
# indicate that a word is a conjugated word of a given verb)
LINK = "LINK-"


def listify(f):
    @functools.wraps(f)
    def listify_helper(*args, **kwargs):
        return list(f(*args, **kwargs))
    return listify_helper


def parse_xml(filename):
    """Extract all information related to a Wiktionary page without
    loading the whole XML file in memory.

    Only the title and the wikitext of the page are extracted.
    """

    # using interparse allow us to avoid loading all xml file into
    # memory
    for event, elem in iterparse(filename, ('start', 'end')):
        # removes namespace
        elem.tag = elem.tag.split('}', 1)[-1]
        if event == 'end' and elem.tag == "page":
            title = elem.find("title").text

            text = elem.find(".//text")
            yield title, text.text

# =================
# Wikitext Parsers
# =================
#
# There is one parser for each language. Each parser takes the page title (i.e.
# the word) and the page content (i.e. the wikitext) and extract POS
# information for them. It returns a list of tuples, each tuple describing a
# word and a POS, which allows to also extract information about the different
# forms of the word that is contained in the page.


def extract_language(text, language_names):
    captured_text = []

    language_names = {"=={}==".format(l) for l in language_names}
    title2 = re.compile("^==([^=]+)==$")

    capture = False
    found_lang = False
    for line in text.split("\n"):

        if line.strip() in language_names:
            capture = True
            found_lang = True
        elif title2.search(line):
            capture = False
            found_lang = True

        if capture:
            captured_text.append(line)

    if found_lang and not captured_text:
        return

    return "\n".join(captured_text)


def expand_fi_subs(what, pos_tag, lem=None):
    if not "table" in what:
        return
    
    for w in what.split("\n"):
        if "{" in w or "}" in w or w.startswith("!"):
            continue

        if w.startswith("<div"):
            continue

        w = w.replace("|-", "")

        if not w:
            continue

        for cell in w.split("||"):
            
            if cell.startswith("=="):
                continue

            for style in ['colspan=".*?"',
                          'style=".*?"',
                          '<sup>.*?</sup>',
                          'rowspan=".*?"']:
                r = regex.compile(style)
                cell = r.sub("", cell)
            
            cell = cell.replace("| ", "")
            cell = cell.strip()

            r = regex.compile("\[\[(.*\|)?(.*)\]\]")
            if r.match(cell):
                rr = r.match(cell)
                cell = rr.group(2)

            cell = cell.replace("'", "").replace("(", "").replace(")", "")
            cell = cell.replace("<br>", ";").replace("<br />+", ";").replace("<br/>", ";").replace("<br />", ";")

            cases = {"nominatiivi", "genetiivi", "partitiivi", "akkusatiivi", "inessiivi","elatiivi", "illatiivi", "adessiivi", "ablatiivi", "allatiivi", "essiivi", "translatiivi", "abessiivi", "instruktiivi", "komitatiivi", "*positiivi:", "*superlatiivi:", "*komparatiivi:"}

            for c in cell.split(";"):

                if c.startswith("+"):
                    continue
                
                for cc in c.split(" "):
                    if cc == "&ndash" or cc in cases:
                        continue
                        
                    if cc.endswith("-") or "[" in cc or not cc or "]" in cc:
                        continue

                    if lem is not None and cc.startswith(lem):
                        yield cc.strip(), pos_tag
                    elif lem is None:
                        yield cc.strip(), pos_tag

        
@listify
def fi_parse_page(text, title):

    if ":" in title and not title.startswith("Liite:"):
        return 

    extract_lang = True

    if title.startswith("Liite:Verbitaivutus/suomi/"):
        pos_tag = "Verbi"
        extract_lang = False

    if title.startswith("Liite:Adjektiivitaivutus/suomi/"):
        pos_tag = "Adjektiivi"
        extract_lang = False

    if extract_lang:
        text = extract_language(text, {"Suomi"})

    if text is None:
        return

    title3 = regex.compile("""
        ===\ ?                 # beginning of title3 markup + optional space
        (                      # list of the titles we are interested in
           Substantiivi
         | Adjektiivi
         | Numeraali
         | Adverbi
         | Konjunktio
         | Substantiivi
         | Interjektio
         | Pronomini
         | Postpositio
         | Partikkeli
         | Verbi
        )
        \ ?===                  # end of the title
    """, regex.VERBOSE)

    if title3.search(text):
        for r in title3.findall(text):
            yield title, r

    # Search all templates in page
    template = regex.compile("\{\{[^{]*?\}\}")
    for template_call in template.findall(text):
        template_name = template_call.split("|")[0][2:].replace("}}", "")

        # Templates the does not contains any flexion (i.e. that must
        # not be expanded)
        templates = {"fi-v-taivm": "Verbi",
                     "fi-verbi": "Verbi",
                     "fi-subs": "Substantiivi",
                     "etunimi": "Substantiivi",
                     "fi-adj": "Adjektiivi",
                     "fi-num-j": "Numeraali",
                     "num-k": "Numeraali",
                     "fi-adv": "Adverbi"}
        try:
            yield title, templates[template_name]
            pos_tag = templates[template_name]
            continue
        except KeyError:
            pass

        if template_name.startswith("fi-v-taivm"):
            yield title, "Verbi"
        
        expanded = requests.get("http://fi.wiktionary.org/w/api.php",
                                params={"action": "expandtemplates",
                                        "prop": "wikitext",
                                        "title": title,
                                        "text": template_call,
                                        "format": "json"}).json()["expandtemplates"]["wikitext"]

        if template_name.startswith("fi-subs"):
            for x in expand_fi_subs(expanded, "Substantiivi"):
                yield x
        elif template_name == "fi-pron":
            for x in expand_fi_subs(expanded, "Pronomini"):
                yield x
        elif template_name.startswith("fi-a-taiv-"):
            try:
                for x in expand_fi_subs(expanded, pos_tag):
                    yield x
            except UnboundLocalError:
                pass
        elif template_name.startswith("fi-verbi-taiv-"):
            lem = template_call.split("|")[1].replace("}", "")
            for x in expand_fi_subs(expanded, pos_tag, lem):
                yield x
                    
@listify
def cs_parse_page(text, title):
    import regex
    import sys

    # This page contains a strange block I did not want to take care of
    if title == "Papua-Nová Guinea":
        return

    # Extract the part of the page that defines the CS word
    # This part starts with a title of level 2 that contains the word čeština
    title2 = re.compile("^==([^=]+)==$")
    czech_text = []

    capture = False
    found_lang = False
    for line in text.split("\n"):
        if line.strip() in {"== čeština ==", "==čeština=="}:
            capture = True
            found_lang = True
        elif title2.search(line):
            capture = False
            found_lang = True

        if capture:
            czech_text.append(line)

    if found_lang and not czech_text:
        return

    if czech_text:
        text = "\n".join(czech_text)

    # Czesh is highly flectional language. The different forms of a word are
    # given in a template defines by the following regexp. Each line of this
    # template defines a form that will be extracted with the `line` regexp
    block = regex.compile("""
    \{\{                    # beginning of template
       (                    # grammatical category
          Adjektivum
        | Substantivum
        | Sloveso
        | Zájmeno
        | Číslovka
        | Číslovka\ adj
        | Zájmeno\ adj
       )\ \(cs\)\\n         # language + start of the block
       ((.|\\n)*?)
    \\n\ *\}\}              # end of template
    """,
                          regex.MULTILINE | regex.VERBOSE)

    line = regex.compile("\| ?[\w1-9]+ *= ?\[?\[?(.+?)\]?\]?( / \[?\[?(.+?)\]?\]?)?$")

    # some POS are also given in a level 3 title

    title3 = regex.compile("""
        ===\ ?                 # beginning of title3 markup + optional space
        (                      # list of the titles we are interested in
           příslovce
         | podstatné\ jméno
         | zkratka
         | sloveso
         | částice
         | číslovka
         | citoslovce
         | spojka
         | předpona
         | přídavné\ jméno
         | slovní\ spojení
         | ve\ složeninách
         | fráze
         | předložka
         | zájmeno
         | zájmeno\ \(1\)
        )
        \ ?===                  # end of the title
    """, regex.VERBOSE)

    if not block.search(text) and not title3.search(text):
        return

    if title3.search(text):
        for r in title3.findall(text):
            yield title, r

    r = block.search(text)
    if not r:
        return
    pos = r.group(1)

    t = r.group(2).strip()

    # empty block
    if t.strip() == "|":
        return

    for l in t.split("\n"):
        l = l.strip()

        if not l:
            continue

        if l == "| nesklonné":
            # uninflected words
            yield title, pos
            continue

        if l.endswith("="):
            # missing form
            continue

        l = line.match(l)
        if not l:
            print("Error when parsing: {}".format(title))
            sys.exit(1)

        word = l.groups()[0]
        if word:
            yield word, pos

        # this happens when a given flexion (e.g. ACC feminin) can have two
        # forms
        word = l.groups()[-1]
        if word:
            yield word, pos


@listify
def id_parse_page(text, title):
    import re
    title3 = re.compile("^===(([^=]+)(=)*([^=]+))===$")
    pos_tag_with_lang = re.compile("\{\{-([_ a-z]+)-\|id\}\}")
    pos_tag_with_lang2 = re.compile("\{\{id-([\w ]+)\}\}")

    for line in text.split("\n"):
        # the regexp only matches Indonesian (cf. the "\|id" part) POS
        # tags --> we do not have to rely on the language identification
        if pos_tag_with_lang.search(line):
            yield title, pos_tag_with_lang.search(line).group(1)

        if title3.match(line):
            yield title, title3.match(line).group(1).strip()

        if pos_tag_with_lang2.search(line):
            yield title, pos_tag_with_lang2.search(line).group(1)

        if line.startswith("{{imbuhan"):
            yield title, line.split("|")[0]


@listify
def de_parse_page(text, title):
    import regex
    template_with_lang = regex.compile("\{\{(([\w ]+)\|)+Deutsch\}\}")
    link_template = regex.compile("\{\{Grundformverweis\|(\D+)\}\}")

    for line in text.split("\n"):
        if template_with_lang.search(line):
            res = template_with_lang.search(line)
            yield title, res.group(1).strip()

        if link_template.match(line):
            res = link_template.match(line)
            yield title, LINK + res.group(1).strip()


def sv_parse_page(text, title):
    import re
    lang_start = re.compile("^\{\{([A-Z]{2})(-[A-Z]{2})?")
    # Language information can also be provided in a title of level 2
    title2 = re.compile("^==([^=]+)==$")
    # POS information are given in title of level 3
    title3 = re.compile("^===([^=]+)===$")

    for line in text.split("\n"):
        # start of language template
        if lang_start.match(line) or title2.match(line):
            res = lang_start.match(line) or title2.match(line)
            capture = res.group(1).strip() in {"Svenska"}

        if title3.match(line) and capture:
            yield title, title3.match(line).group(1).strip()


@listify
def es_parse_page(text, title):
    import re
    lang_start = re.compile("^\{\{([A-Z]{2})(-[A-Z]{2})?")
    # Language information can also be provided in a title of level 2
    title2 = re.compile("^==([^=]+)==$")
    # POS information are given in title of level 3
    title3 = re.compile("^===([^=]+)===$")

    capture = False
    for line in text.split("\n"):
        line = line.strip()
#        print(repr(line), lang_start.match(line), title3.match(line))

        # start of language template
        if lang_start.match(line) or title2.match(line):
            res = lang_start.match(line) or title2.match(line)
            capture = res.group(1).strip() in {"ES", "Español"}

        if title3.match(line) and capture:
            yield title, title3.match(line).group(1).strip()


@listify
def fr_parse_page(text, title):
    # recognize template that indicates the language of the entry
    # templates are made according to the following pattern:
    # - {{ES|mes}}
    # - {{FR-ES|mes}}
    # - {{SV-ES|mes}}
    # - {{ES}}
    lang_start = re.compile("^\{\{([A-Z]{2})(-[A-Z]{2})?")
    # Language information can also be provided in a title of level 2
    title2 = re.compile("^==([^=]+)==$")
    # POS information are given in title of level 3; some of POS
    # information include a =
    title3 = re.compile("^===(([^=]+)(=)*([^=]+))===$")
    # Flexions are given using a special template
    flex = re.compile("\{\{-flex(.*)\|fr\}\}")
    # Attention à ne pas oublier l'espace dans le nom de la fonction
    # grammaticale !
    pos_tag = re.compile("\{\{S\|([\w ]*)\|fr(\|num=\d+)?\}\}", re.UNICODE)

    capture = False
    lang_found = False
    for line in text.split("\n"):
        # start of language template
        if lang_start.match(line) or title2.match(line):
            res = lang_start.match(line) or title2.match(line)
            # XXX check if the second element really happens/is correct
            capture = res.group(1).strip() in {"{{langue|fr}}"}
            lang_found = True

        if pos_tag.search(line) and capture:
            yield title, pos_tag.search(line).group(1)

        if title3.match(line) and capture:
            yield title, title3.match(line).group(1).strip()

        if flex.match(line) and capture:
            yield title, line.strip()

    if not lang_found:
        pass
        # this happens, for instance, in page that only contains a "REDIRECT"
        # r = "REDIRECT" in text
        # logging.warning("no lang found in {} (REDIRECT={})".format(title, r))


@listify
def it_parse_page(text, title):
    import re
    lang_start = re.compile("^\{\{([A-Z]{2})(-[A-Z]{2})?")
    title2 = re.compile("^==([^=]+)==$")
    title3 = re.compile("^===(([^=]+)(=)*([^=]+))===$")
    pos_tag = re.compile("\{\{-([ a-z]+)-\}\}")
    pos_tag_with_lang = re.compile("\{\{-([ a-z]+)-\|it\}\}")

    capture = False
    lang_found = False
    for line in text.split("\n"):
        # start of language template
        if lang_start.match(line) or title2.match(line):
            res = lang_start.match(line) or title2.match(line)
            capture = res.group(1).strip() in {"{{-it-}}"}
            lang_found = True

        # the regexp only matches italian (cf. the "\|it" part) POS
        # tags --> we do not have to rely on the language identification
        if pos_tag.search(line) and capture:
            yield title, pos_tag.search(line).group(1)

        if pos_tag_with_lang.search(line):
            yield title, pos_tag_with_lang.search(line).group(1)

        if title3.match(line) and capture:
            yield title, title3.match(line).group(1).strip()

    if not lang_found:
        pass
        # this happens, for instance, in page that only contains a "REDIRECT"
        # r = "REDIRECT" in text
        # logging.warning("no lang found in {} (REDIRECT={})".format(title, r))


@listify
def el_parse_page(text, title):
    import regex

    lang_start = regex.compile("^==\{\{-([a-z][a-z])-\}\}==")

    template = regex.compile("\{\{([\w ]+)\}\}")
    template_with_lang = regex.compile("\{\{([\w ]+)\|el\}\}")

    capture = False
    for line in text.split("\n"):
        # start of language template
        if lang_start.match(line):
            res = lang_start.match(line)
            capture = res.group(1) == "el"

        if "μεταφράσεις" in line:
            capture = False

        # the regexp only matches greek (cf. the "\|el" part) POS
        # tags --> we do not have to rely on the language identification
        if template.search(line) and capture:
            yield title, template.search(line).group(1)

        if template_with_lang.search(line):
            yield title, template_with_lang.search(line).group(1)


@listify
def ro_parse_page(text, title):
    text = extract_language(text, {"{{limba|ron}}", "{{limba|ro}}"})
    if not text:
        return

    block = regex.compile("""
    \{\{                    # beginning of template
       (                    # grammatical category
          adjectiv
        | substantiv
        | verb
        | numeral
       )-ron\ *\\n         # language + start of the block
       ( (\| [^\|\\n]* \\n)* )
    \\n?\ *\}\}              # end of template
    """,
    regex.MULTILINE | regex.VERBOSE)

    line = regex.compile("\| ?[\w1-9-]+ *= ?\[?\[?(.+?)\]?\]?$")

    template = regex.compile("""
        \{\{\-                 # beginning of template
        (                      # grammatical category
           abreviere
         | adverb
         | articol
         | conjuncție
         | comp
         | expresie
         | interjecție
         | locuțiune
         | numeral
         | prefix
         | prepoziție
         | pronume
         | sufix
         | verb
         | adjectiv
         | substantiv
        )
        -\|ron\}\}                  # end of template
    """, regex.VERBOSE)

    if template.search(text):
        for r in template.findall(text):
            if "lipsă" in r:
                continue
            yield title, r

    # FIXME: what if multiple blocks?

    r = block.search(text)
    if not r:
        return
    pos = r.group(1)

    t = r.group(2).strip()

    # empty block
    if t.strip() == "|":
        return

    for l in t.split("\n"):
        l = l.strip()

        if not l:
            continue

        if l.endswith("="):
            # missing form
            continue

        l = line.match(l)
        if not l:
            import sys
            print("Error when parsing: {}".format(title))
            sys.exit(1)

        word = l.groups()[0]
        if word:
            if regex.compile("^\{\{.*\}\}$").match(word):
                continue
            if regex.compile("^\(.*\)$").match(word):
                # FIXME: what about "manifesta"?
                continue
            yield word, pos


def read_mapping(mapping_filename):

    mapping = defaultdict(list)

    for line in open(mapping_filename, "rt"):
        line = line.strip()

        if line == "":
            continue

        if line.startswith("#"):
            continue

        if line.startswith("--"):
            cat = line.replace("-", "")
            continue

        label = " ".join(line.split(" ")[1:])
        mapping[label].append(cat)

    return mapping


def filter_pages(articles, lang):

    parse_page = eval("{}_parse_page".format(lang))
    for title, text in articles:
        # pages with a column in their title correspond to special pages such
        # as "MediaWiki:Aboutpage" or "Πρότυπο:Sitesupportpage"
        # if ":" in title or title in {"Main Page", "Pagina principale"}:
        #     continue

        if text is None:
            continue

        pos = parse_page(text, title)

        if not pos:
            continue

        for word, p in pos:
            if "/" in word:
                yield word.split("/")[0], [p]
                yield word.split("/")[1], [p]
            else:
                yield word, [p]


def unlink(c, constraints):
            """
            Resolve links in Wiktionary entries
            """
            if not c.startswith(LINK):
                return [c]

            word = c.replace(LINK, "")

            if "|" in word:
                word = [w for w in word.split("|")
                        if "=" not in w and "#" not in w]
                word = word[0]

            return constraints[word] if word in constraints else []


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--wiki", required=True,
                        help="filename of the wiktionary dump")
    parser.add_argument("--lang", required=True,
                        choices=["fr", "el", "it", "es",
                                 "de", "sv", "id", "cs",
                                 "fi","ro"])
    parser.add_argument("--map",
                        required=True,
                        help="filename of the mapping between wiktionary "
                        " and universal POS")
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    articles_stream = parse_xml(args.wiki)
    pos_stream = filter_pages(articles_stream, lang=args.lang)

    mapping = read_mapping(args.map)

    constraints = defaultdict(set)

    pos_map = lambda p: mapping[p] if not p.startswith(LINK) else [p]

    for word, pos in pos_stream:
        print(word)
        try:
            pos = (p.strip() for p in pos)
            c = map(pos_map, pos)
            c = {p for p in chain.from_iterable(c) if p != "IGNORE"}
            constraints[word].update(c)
        except KeyError as e:
            if str(e) == "' '":
                # This is happenning once in greek; I am not sure it is
                # relevant at all
                logging.warning("empty pos in page '{}'".format(word))
                continue
            print("== Error ==")
            print(word, e, pos)
            import sys
            sys.exit(1)

    u = partial(unlink, constraints=constraints)
    # unlink twice to resovle all links in German
    if args.lang == "de":
        constraints = {w: {c for c in chain.from_iterable(map(u, constraints[w]))}
                       for w in constraints}
        u = partial(unlink, constraints=constraints)
        constraints = {w: {c for c in chain.from_iterable(map(u, constraints[w]))}
                       for w in constraints}

    with open(args.output, "wt") as ofile:
        for key, items in constraints.items():
            for el in items:
                ofile.write("{}\t{}\n".format(key.replace(" ", "_"), el))
