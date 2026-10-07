"""Señales lingüísticas (español + inglés) y prototipos semánticos.

Las expresiones regulares son SOLO una señal más: el peso principal del
detector está en la similitud semántica con los prototipos (que funciona en
cualquier idioma gracias al modelo multilingüe), la estructura del tema y la
prosodia. Encontrar "increíble" no convierte una frase en un buen momento.
"""
from __future__ import annotations

import re


def _rx(*patterns: str) -> re.Pattern:
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)


# Frases que abren curiosidad o plantean algo fuerte.
HOOK = _rx(
    r"\b(sab[eé]s|sab[ií]as|sabes) (qu[eé]|por qu[eé]|lo que|cu[aá]l)",
    r"\bte (imagin[aá]s|imaginas)\b", r"\bimagin[aá]te\b",
    r"\blo que (nadie|pocos|muy pocos|casi nadie)\b",
    r"\b(el|un|los) secretos?\b", r"\bla verdad (es|sobre|de)\b",
    r"\b(el|la|lo) (mayor|peor|mejor|m[aá]s (grande|importante|dif[ií]cil))\b",
    r"\bte (voy|vamos) a (contar|decir|explicar|mostrar)\b",
    r"\b(nadie|todo el mundo|la mayor[ií]a|la gente) (te )?(sabe|cree|piensa|dice|habla)\b",
    r"\bnunca (te |me |nos |le )?(cont[eé]|dije|imagin[eé]|pens[eé])\b",
    r"\b(qu[eé]|c[oó]mo) (es posible|pas[oó]|hiciste|hice|lograste|logr[eé])\b",
    r"\bpor qu[eé]\b", r"\bqu[eé] pasar[ií]a si\b", r"\b(error|errores|mentira|problema)\b",
    r"\bdid you know\b", r"\bnobody (tells|talks)\b", r"\bthe (truth|secret|biggest|worst|best)\b",
    r"\blet me tell you\b", r"\bhere'?s the thing\b", r"\bwhat if\b", r"\bwhy\b",
)

# Comienzos que dependen de algo dicho antes (malos inicios de clip).
DEPENDENT_START = _rx(
    r"^(y|e|pero|entonces|o sea|porque|que|pues|as[ií] que|aunque|sino|adem[aá]s|tambi[eé]n|tampoco|o|ni)\b",
    r"^(eso|esto|esa|ese|esos|esas|aquello|ah[ií]|all[ií])\b",
    # Respuestas a algo que dijo otro ("No, no, contame…", "Claro, …", "Exacto.").
    r"^(no|s[ií]|claro|exacto|obvio|tal cual|bien|ah|ok|dale|seguro|totalmente|correcto)[,.!]",
    r"^(él|ella|ellos|ellas|lo mismo|por eso|en ese (momento|caso))\b",
    r"\bcomo (te |les |lo |ya )?(dije|dec[ií]a|coment[eé]|mencion[eé]|hablamos|habl[aá]bamos)\b",
    r"\blo que (te |les )?(dec[ií]a|dije|comentaba|hablamos)\b", r"\bvolviendo a\b",
    r"^(and|but|so|because|or|then|also)\b", r"^(that|this|it|he|she|they)\b",
    r"\b(as|like) i (said|mentioned)\b",
)

# Referencias hacia atrás dentro del clip (restan autonomía aunque no estén al inicio).
BACK_REFERENCE = _rx(
    r"\bcomo (te |les |ya )?(dije|dec[ií]a|coment[eé]|mencion[eé])\b", r"\b(lo|eso) que (dijimos|hablamos)\b",
    r"\b(hace un rato|m[aá]s temprano|antes) (te |les )?(dije|coment[eé]|hablamos)\b",
    r"\bas i (said|mentioned)\b", r"\bearlier\b",
)

# Cierres: conclusión, lección, remate.
PAYOFF = _rx(
    r"\bpor eso\b", r"\bas[ií] que\b", r"\ben (resumen|definitiva|conclusi[oó]n)\b", r"\bal final\b",
    r"\bla (clave|lecci[oó]n|moraleja|conclusi[oó]n)\b", r"\blo (importante|que aprend[ií])\b",
    r"\baprend[ií]\b", r"\b(y )?as[ií] (fue|es) como\b", r"\bde eso se trata\b", r"\bnunca m[aá]s\b",
    r"\bthat'?s why\b", r"\bthe (point|lesson|key) is\b", r"\bbottom line\b", r"\bin the end\b", r"\bi learned\b",
)

# Información útil: consejos, explicaciones, datos.
VALUE = _rx(
    r"\b(consejo|recomiendo|recomendaci[oó]n|tip)\b", r"\b(ten[eé]s|tienes|hay|tenemos) que\b",
    r"\b(la clave|el truco|el secreto|la f[oó]rmula|el m[eé]todo|la estrategia)\b",
    r"\bpor ejemplo\b", r"\bes decir\b", r"\bsignifica\b", r"\bfunciona\b", r"\bpaso (uno|dos|1|2)\b",
    r"\b(\d+|mil|millones?|cien|por ciento|%)\b", r"\b(d[oó]lares|pesos|euros)\b",
    r"\b(you (have|need) to|for example|the key|means|works)\b",
)

# Narrativa: historias personales y relatos.
STORY = _rx(
    r"\buna vez\b", r"\bun d[ií]a\b", r"\b(me )?acuerdo\b", r"\brecuerdo (que|cuando)\b",
    r"\bcuando (yo )?(era|ten[ií]a|empec[eé]|llegu[eé]|estaba)\b", r"\bresulta que\b", r"\bde repente\b",
    r"\ben ese momento\b", r"\bme pas[oó]\b", r"\blo que pas[oó]\b", r"\bdecid[ií]\b", r"\bempec[eé]\b",
    r"\bone day\b", r"\bi remember\b", r"\bwhen i was\b", r"\bit turns out\b", r"\bsuddenly\b",
)

# El que habla pasa a otro tema: un clip no debería terminar acá.
TRANSITION = _rx(
    r"(otro|un) tem(a|ita)", r"cambiando de tema", r"(pasemos|vamos) a (hablar|otro)",
    r"te (quer[ií]a|quiero) preguntar", r"otra pregunta", r"hablemos de",
    r"abordar (un|otro|el)", r"para (ir )?(cerrar|terminar)", r"antes de (cerrar|terminar)",
    r"(moving on|another (topic|question)|let'?s talk about|i wanted to ask)",
)

# Logística del programa/canal: pausas, saludos, cortesías, autopromoción.
# Ningún Short debería contener esto.
META = _rx(
    r"\b(ya )?(regresamos|volvemos)\b", r"\bno se vayan\b", r"\bestamos de (regreso|vuelta)\b",
    r"\b(la|una) pausa\b", r"\bcorte comercial\b", r"\bbienvenid[oa]s? (a|al)\b",
    r"\bgracias por (invitarme|estar|venir|acompañarnos|ver)\b", r"\bun (placer|gusto) (estar|tenerte)\b",
    r"\bsuscr[ií]b", r"\b(dale|den|denle) (like|me gusta)\b", r"\b(link|enlace) en la descripci[oó]n\b",
    r"\bpatrocin", r"\bnos vemos en (el|la) pr[oó]xim", r"\bhasta la pr[oó]xima\b",
    r"\b(we'?ll be right back|after the break|subscribe|link in the description|thanks for having me)\b",
)

# Emoción, sorpresa, conflicto.
IMPACT = _rx(
    r"\bincre[ií]ble\b", r"\bimpresionante\b", r"\bno (lo )?(pod[ií]a|puedo) creer\b", r"\bloco\b",
    r"\b(brutal|terrible|horrible|tremendo|dur[ií]simo|fuert[ií]simo)\b", r"\b(llor[eéa]|llorando|l[aá]grimas)\b",
    r"\b(miedo|p[aá]nico|angustia|verg[uü]enza)\b", r"\b(fracas[eéo]|quebr[eéa]|perd[ií] todo|me ech[aá]ron)\b",
    r"\b(muri[oó]|muerte|enfermedad|c[aá]ncer)\b", r"\b(odio|amo|me encanta)\b", r"\b(mentira|injusto|estafa)\b",
    r"\b(amazing|insane|crazy|unbelievable|terrible|scared|cried)\b",
)

LAUGH = _rx(r"\bja(ja)+\b", r"\(risas?\)", r"\[risas?\]", r"\b(ha){2,}\b", r"\(laughs?\)", r"\[laughter\]")

# Opiniones firmes y absolutos.
OPINION = _rx(
    r"\b(yo )?(creo|pienso|siento) que\b", r"\bestoy (convencid[oa]|segur[oa])\b", r"\ben realidad\b",
    r"\b(sinceramente|honestamente|la verdad)\b", r"\bno estoy de acuerdo\b", r"\bel problema (es|de)\b",
    r"\b(nunca|jam[aá]s|siempre|nadie|todo el mundo)\b", r"\b(i (think|believe)|honestly|the problem is)\b",
)

# Muletillas.
FILLER_WORDS = {"eh", "ehm", "em", "emm", "mm", "mmm", "uh", "um", "uhm", "ah", "este", "osea"}
FILLER_PHRASES = _rx(r"\bo sea\b", r"\bdigamos\b", r"\b¿?viste\??", r"\b¿?no\?", r"\byou know\b", r"\bkind of\b")
# Palabras a recortar al inicio de un clip si van seguidas de pausa/coma ("Bueno, ...", "Eh, ...").
LEADING_TRIM = {"eh", "ehm", "em", "emm", "mm", "mmm", "uh", "um", "uhm", "ah", "bueno", "pues", "este", "so", "well"}

# Final de frase que deja algo colgando.
DANGLING_END = _rx(
    r"\b(y|e|pero|que|de|del|la|el|los|las|un|una|en|con|por|para|porque|o|como|a)[\s,]*$",
    r",\s*$", r"\b(and|but|the|of|to|that|because)[\s,]*$",
)

NUMBER = re.compile(r"\b\d+([.,]\d+)?\b|%|\b(mil|millones?|cien|doscientos|trescientos|quinientos)\b", re.IGNORECASE)


# --------------------------------------------------------------------------
# Prototipos semánticos. El modelo de embeddings es multilingüe, así que
# estos ejemplos también sirven para videos en otros idiomas.
# --------------------------------------------------------------------------
PROTOTYPES: dict[str, list[str]] = {
    "hook": [
        "¿Sabés cuál fue el peor error que cometí en mi vida?",
        "Esto es algo que casi nadie te cuenta.",
        "Lo que pasó después fue increíble.",
        "Te voy a contar algo que me cambió la vida.",
        "La mayoría de la gente cree esto, y está completamente equivocada.",
        "Nunca imaginé que eso me iba a pasar a mí.",
        "¿Por qué casi todos los que lo intentan terminan fracasando?",
        "Hay una cosa que nadie te dice cuando empezás.",
        "El secreto es mucho más simple de lo que pensás.",
        "Ese día lo perdí todo.",
        "Me pasó algo que me marcó para siempre.",
        "Para mí es mentira eso que dicen de que la gente no cambia.",
        "Here's something nobody tells you.",
        "What happened next completely changed my life.",
    ],
    "payoff": [
        "Y por eso hoy pienso que lo más importante es no rendirse.",
        "Esa fue la lección más grande que aprendí.",
        "Al final, todo se trata de animarse a empezar.",
        "Así que mi consejo es que lo hagas igual, aunque tengas miedo.",
        "Y así fue como terminé haciendo lo que hago hoy.",
        "Eso es lo que marca la diferencia entre los que lo logran y los que no.",
        "Nunca más volví a cometer ese error.",
        "That's why I always say you have to try.",
        "And that was the moment everything changed.",
    ],
    "story": [
        "Cuando tenía veinte años me fui de mi casa sin nada.",
        "Me acuerdo que un día llegué al trabajo y me despidieron.",
        "Resulta que en ese momento no tenía ni un peso.",
        "Mi papá me dijo algo que nunca voy a olvidar.",
        "Empecé vendiendo en la calle y de a poco fui creciendo.",
        "Un día recibí una llamada que me cambió todo.",
        "When I was a kid, my family had nothing.",
        "One day I decided to quit my job and start over.",
    ],
    "value": [
        "El consejo que le daría a cualquiera que quiera empezar es este.",
        "Hay tres cosas que tenés que hacer para que funcione.",
        "La clave está en hacerlo todos los días, aunque sea poco.",
        "Por ejemplo, si ahorrás el diez por ciento de lo que ganás, en cinco años cambia todo.",
        "Esto funciona porque el cerebro se acostumbra a la rutina.",
        "Lo que tenés que entender es cómo funciona realmente el dinero.",
        "The most important thing you can do is start before you feel ready.",
        "Here's exactly how it works, step by step.",
    ],
    "impact": [
        "Fue el momento más duro de toda mi vida.",
        "No lo podía creer, me quedé helado.",
        "Lloré como nunca en mi vida.",
        "Me dio muchísima vergüenza, fue terrible.",
        "Me reí tanto que no podía parar.",
        "Es una locura lo que pasó, nadie lo esperaba.",
        "Estoy totalmente en contra de eso, me parece una mentira.",
        "Tenía tanto miedo que no podía dormir.",
        "Me quedé en blanco frente a todos y no supe qué decir.",
        "Fue el peor fracaso de mi vida y me costó años superarlo.",
        "Sufrí muchísimo, fue una lucha muy personal.",
        "Me sentí humillado delante de todo el mundo.",
        "It was the hardest moment of my entire life.",
        "I couldn't believe what I was seeing.",
    ],
    "meta": [
        "Ya regresamos después de la pausa, no se vayan.",
        "Bienvenidos a un nuevo programa, qué bueno tenerte aquí.",
        "Muchas gracias por invitarme, es un placer estar aquí con ustedes.",
        "Suscribite al canal y dale like si te gustó el video.",
        "Este episodio es patrocinado por nuestra marca amiga.",
        "Bueno, se nos terminó el tiempo, nos vemos la próxima semana.",
        "We'll be right back after the break.",
        "Thanks for having me on the show.",
    ],
    "dependent": [
        "Como te decía antes, eso es lo que pasa.",
        "Y eso que estábamos hablando recién.",
        "Volviendo a lo que dijiste hace un rato.",
        "Sí, exacto, eso mismo.",
        "Claro, claro, eso.",
        "Y bueno, lo otro también.",
        "As I was saying before.",
        "Yeah, exactly, that one.",
    ],
}
