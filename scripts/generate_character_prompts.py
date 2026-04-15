#!/usr/bin/env python3
"""Generate diverse, self-contained character prompts for Gemini/Sora.

Each prompt is fully standalone — specific style, specific character, specific
pose. Format matches the reference prompt with demographic descriptor and all
framing/background specifications.

Usage::

    python3 scripts/generate_character_prompts.py --count 100 --start_id 61
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Pools
# ---------------------------------------------------------------------------

STYLES = [
    "Full-body watercolor",
    "Full-body oil painting",
    "Full-body gouache illustration",
    "Full-body digital painting",
    "Full-body anime cel-shaded illustration",
    "Full-body manga ink drawing",
    "Full-body flat vector illustration",
    "Full-body retro pixel art",
    "Full-body storybook illustration in gouache",
    "Full-body concept art painting",
    "Full-body pop art illustration",
    "Full-body pastel chalk illustration",
    "Full-body ink and wash illustration",
    "Full-body soft-shaded cartoon",
    "Full-body minimal line art with watercolor wash",
]

STYLE_FINISHES = {
    "Full-body watercolor": "Delicate watercolor.",
    "Full-body oil painting": "Rich oil painting with visible brushwork.",
    "Full-body gouache illustration": "Warm gouache tones.",
    "Full-body digital painting": "Painterly digital art.",
    "Full-body anime cel-shaded illustration": "Clean anime cel shading.",
    "Full-body manga ink drawing": "High-contrast ink lines.",
    "Full-body flat vector illustration": "Bold flat colors with minimal shading.",
    "Full-body retro pixel art": "Crisp retro pixel art.",
    "Full-body storybook illustration in gouache": "Cozy children's book style.",
    "Full-body concept art painting": "Detailed concept art.",
    "Full-body pop art illustration": "Bold pop art with graphic shapes.",
    "Full-body pastel chalk illustration": "Soft pastel chalk textures.",
    "Full-body ink and wash illustration": "Traditional ink and wash.",
    "Full-body soft-shaded cartoon": "Smooth cartoon shading.",
    "Full-body minimal line art with watercolor wash": "Clean linework with soft color wash.",
}

POSES = [
    ("standing in a neutral rest pose", "neutral"),
    ("standing in an A-pose with arms slightly out and legs apart", "a_pose"),
    ("standing in a T-pose with arms fully extended horizontally", "t_pose"),
    ("walking with one foot forward in a confident stride", "walking"),
    ("mid-swing of a powerful action pose", "action"),
    ("crouched low with one hand on the ground", "crouch"),
    ("sitting on the floor with legs crossed", "sitting"),
    ("standing with hands on hips in a heroic pose", "hands_on_hips"),
    ("standing with one arm raised overhead", "arm_raised"),
    ("leaning slightly forward in an expressive gesture", "leaning"),
    ("holding an object close to the chest with both hands", "holding_chest"),
    ("stepping forward in an enthusiastic stance", "stepping"),
    ("mid-run with arms pumping", "running"),
    ("mid-jump with both feet off the ground", "jumping"),
    ("kneeling on one knee in a reverent pose", "kneeling"),
    ("standing side-profile with arms crossed", "arms_crossed"),
    ("dancing with one leg lifted and arms flowing", "dancing"),
    ("reaching forward with one outstretched hand", "reaching"),
]

# (character description, demographic descriptor)
# description = visible outfit + props; demographic = gender/ethnicity/feel
CHARACTERS = [
    ("a teenage girl in a sailor school uniform — pleated skirt, navy top with white collar, ribbon tie, knee socks, loafers, and a schoolbag slung over one shoulder",
     "feminine-presenting, cheerful, with East Asian features"),
    ("an elderly wizard with a long white beard, dark blue robe embroidered with silver moons, a tall pointed hat, leather sandals, and a twisted wooden staff",
     "masculine-presenting, wise, with pale European features"),
    ("a muscular barbarian warrior in a fur loincloth, leather boots, wrist bracers, with tribal tattoos, and wielding a massive double-bladed axe",
     "masculine-presenting, fierce, with bronzed Nordic features"),
    ("a bicycle delivery rider in a branded windbreaker, cap, cargo shorts, sneakers, a backpack, and holding a pizza box",
     "non-binary-presenting, energetic, with South Asian features"),
    ("a noir detective in a long trench coat, fedora, dress shirt, slacks, leather shoes, and a cigarette held loosely",
     "masculine-presenting, brooding, with Mediterranean features"),
    ("a tiny hedgehog knight standing upright in handmade tin-plate armor, tiny gauntlets, a twig sword, and a leaf cape",
     "androgynous creature character, plucky and small"),
    ("a space marine in bulky power armor with shoulder pauldrons, full-face helmet, a plasma rifle slung across the chest, and heavy armored boots",
     "masculine-presenting, imposing, visible armor only"),
    ("a high school baseball player in a white jersey, striped pants, cleats, batting gloves, helmet, and gripping a wooden bat",
     "masculine-presenting, athletic, with East Asian features"),
    ("a 1960s pop singer in a bright mod mini-dress, white go-go boots, big hoop earrings, a beehive hairdo, and a microphone in one hand",
     "feminine-presenting, vibrant, with Black features"),
    ("a yoga instructor in fitted leggings, a tank top, bare feet, a beaded bracelet, and a rolled mat under one arm",
     "feminine-presenting, serene, with South American features"),
    ("a dark elf ranger in dark leather armor, a hooded cloak, twin curved daggers, tall boots, and long silver-white hair",
     "androgynous, mysterious, with pointed ears and deep purple skin"),
    ("a chubby baker in a white apron over a checkered shirt, a chef's hat, rolled sleeves, wooden clogs, and a tray of pastries",
     "masculine-presenting, friendly, with Northern European features"),
    ("a Renaissance courtier in a velvet doublet, puffed hose, a ruff collar, a feathered cap, pointed shoes, and a rapier at the hip",
     "masculine-presenting, refined, with pale European features"),
    ("a magical girl in a frilly pink dress with layered skirts, thigh-high white boots, puff sleeves, twin ponytails, and wielding a star-tipped wand",
     "feminine-presenting, sparkling, with fair skin and large anime eyes"),
    ("a hiker in cargo shorts, trail boots, a flannel shirt tied at the waist, a sun hat, and carrying a heavy backpack",
     "masculine-presenting, rugged, with Latin features"),
    ("a steampunk inventor in a brass-buckled corset, a leather utility skirt, knee-high laced boots, goggles pushed up on the forehead, and carrying a wrench",
     "feminine-presenting, inventive, with ginger hair and freckled pale skin"),
    ("a skateboarder teen in baggy jeans, an oversized graphic t-shirt, high-top sneakers, a backwards cap, and wrist guards",
     "masculine-presenting, laid-back, with mixed-race features"),
    ("a gardener grandmother in a floral apron, a wide-brimmed straw hat, rubber clogs, garden gloves, and carrying a watering can",
     "feminine-presenting, kind, with Southeast Asian features"),
    ("a ballet dancer in a white tutu, pink pointe shoes, a tight bun, and wrist ribbons, with defined dancer's posture",
     "feminine-presenting, graceful, with Eastern European features"),
    ("a small dragon boy with green scales, tiny wings, a lizard tail, wearing a simple brown tunic, rope belt, and barefoot",
     "androgynous young creature character, curious and bright"),
    ("an 8-bit-style plumber in blue denim overalls, a red cap, white gloves, brown work boots, and a mustache",
     "masculine-presenting, plucky, with Italian features"),
    ("a samurai in a full kimono and hakama, two katanas at the belt, straw sandals, a topknot hairstyle, and a stern expression",
     "masculine-presenting, disciplined, with Japanese features"),
    ("a 1970s disco dancer in flared bell-bottoms, a rhinestone vest, platform shoes, layered gold chains, and an impressive afro",
     "masculine-presenting, charismatic, with Black features"),
    ("a programmer in a hoodie, skinny jeans, low sneakers, wireless headphones around the neck, and holding a slim laptop",
     "non-binary-presenting, focused, with Middle Eastern features"),
    ("a desert cowboy in a long duster coat, a wide-brimmed hat, a neck bandana, leather gloves, cowboy boots with spurs, and a revolver holstered",
     "masculine-presenting, weathered, with Mexican features"),
    ("a cat character standing upright on hind legs in a chef's hat and apron, a wooden spatula in one paw, and a mischievous grin",
     "androgynous cat character, cheerful cartoon"),
    ("a Victorian governess in a long dark dress with a high collar, a small hat, button-up boots, carrying a book",
     "feminine-presenting, stern, with English features"),
    ("a cyber-ninja in a sleek black bodysuit with glowing neon accents, a mask covering the lower face, wrist gauntlets, and a katana drawn",
     "androgynous, sleek, with mirrored visor hiding the eyes"),
    ("a mermaid princess with an iridescent fish tail, a shell bikini top, a pearl necklace, flowing long hair, and seaweed hair ornaments",
     "feminine-presenting, ethereal, with Polynesian features"),
    ("a fantasy paladin in ornate gold-trimmed plate armor over a tabard, steel greaves, a massive two-handed sword held vertical, and a shield slung on the back",
     "masculine-presenting, noble, with Norse features"),
    ("a cozy winter character in a long puffer coat, a chunky knit scarf, a wool beanie, snow boots, mittens, and holding a steaming coffee cup",
     "feminine-presenting, cozy, with Scandinavian features"),
    ("a farmer in denim overalls over a plaid shirt, work boots, a straw hat, rolled sleeves, and carrying a pitchfork",
     "masculine-presenting, hardy, with Appalachian features"),
    ("a circus clown in big red shoes, striped baggy pants, colorful suspenders, a polka-dot bow tie, red nose, and a curly rainbow wig",
     "androgynous, whimsical, with white face paint"),
    ("a tiny mouse character standing upright in a little dress and apron, tiny leather shoes, whiskers, and carrying a basket of berries",
     "androgynous mouse character, gentle and small"),
    ("a medieval archer in a forest-green tunic, brown leggings, leather boots, a quiver of arrows across the back, and holding a longbow",
     "feminine-presenting, agile, with Celtic features"),
    ("a track sprinter in a bright singlet and shorts, sleek running shoes, numbered bib, and defined runner's build",
     "masculine-presenting, fast, with Kenyan features"),
    ("a 1950s diner waitress in a pink uniform dress, white apron, a small cap, roller skates, and balancing a tray of milkshakes",
     "feminine-presenting, friendly, with pale American features"),
    ("a tall fashion model in a long tailored coat, wide-leg trousers, sharp heeled boots, designer sunglasses, and a sleek bob haircut",
     "feminine-presenting, poised, with West African features"),
    ("a frost mage in layered robes with fur trim, a crystal-tipped staff, leather boots, long braided hair, and ice crystals forming around the hands",
     "feminine-presenting, cold and regal, with Inuit features"),
    ("a chibi astronaut in a puffy white space suit, helmet visor raised, bulky gloves, moon boots, and a small oxygen pack",
     "androgynous small chibi character, curious"),
    ("a pirate captain in a tricorn hat, a long gold-trimmed coat, a white ruffled shirt, a cutlass sheathed at the belt, and knee-high boots",
     "masculine-presenting, roguish, with Caribbean features"),
    ("a battle robot girl with chrome-plated mecha limbs showing joint lines, a human-looking face, short combat skirt, and glowing joint lights",
     "feminine-presenting mecha character, visible hybrid body"),
    ("a ghost child in a long white nightgown, bare feet not quite touching the ground, translucent pale skin, holding a teddy bear",
     "androgynous, melancholic, with translucent form"),
    ("a desert post-apocalyptic raider in layered sand-colored robes, a gas mask, dust goggles, heavy combat boots, and a slung rifle",
     "masculine-presenting, grim, features mostly hidden by mask"),
    ("a blacksmith apprentice in a heavy leather apron over a simple shirt, thick elbow-length gloves, work boots, rolled sleeves, and holding a glowing horseshoe with tongs",
     "masculine-presenting, strong, with Mediterranean features"),
    ("a circus acrobat in a sparkly sequined leotard, soft ballet slippers, wrist ribbons, a tiara, and striking a balletic pose",
     "feminine-presenting, lithe, with Russian features"),
    ("a wise tortoise elder standing upright in a knitted wool vest, a monocle, holding a walking stick with a gnarled grip",
     "androgynous reptile character, scholarly"),
    ("a cyberpunk netrunner in a long black coat with LED lining, a visor over the eyes, combat boots, fingerless gloves, and holding a data-slate",
     "androgynous, sleek, with partially shaved head"),
    ("an ice skater in a sparkling pale-blue costume, white figure skates, delicate tights, sequins catching the light, and arms in graceful sweep",
     "feminine-presenting, elegant, with pale Slavic features"),
    ("a superhero in a bright primary-red-and-blue costume, a flowing cape, domino mask, knee-high boots, and gold belt buckle",
     "masculine-presenting, heroic, with classic comic-book features"),
    ("a yoga practitioner in loose linen pants and a cropped top, barefoot with a mala bracelet, serene expression, and toned arms",
     "feminine-presenting, serene, with Indian features"),
    ("a swamp witch in a layered patchwork cloak of mossy fabrics, tangled gray hair with small bones, bare dirty feet, and a gnarled staff",
     "feminine-presenting, eerie, with pale warty skin"),
    ("a penguin butler standing upright in a tiny tailored tuxedo, a bow tie, white gloves, and balancing a silver serving tray",
     "androgynous penguin character, dignified"),
    ("a Regency-era gentleman in a high-collared tailcoat, a crisp cravat, high-waisted trousers, riding boots, and a cane held in one hand",
     "masculine-presenting, elegant, with English features"),
    ("a shrine maiden in a white kimono-style haori top, bright red hakama pants, wooden geta sandals, holding a paper ofuda charm",
     "feminine-presenting, calm, with Japanese features"),
    ("a forest druid in green moss-woven robes, a crown of branching antlers, bare feet wrapped in vines, holding a small glowing seed",
     "androgynous, ancient, with olive skin and leaf-markings"),
    ("a space bounty hunter in segmented armor plates over a dark bodysuit, a helmet with a glowing orange visor, knee-high boots, and a plasma rifle",
     "gender-ambiguous due to full armor, imposing"),
    ("a national park ranger in a beige uniform shirt with patches, olive-green pants, a wide belt, hiking boots, and a wide-brim ranger hat",
     "masculine-presenting, friendly, with Native American features"),
    ("a lighthouse keeper in a thick wool sweater, waterproof overalls, tall rubber boots, a weathered captain's cap, and holding a lantern",
     "masculine-presenting, weathered, with Irish features"),
    ("a Thai Muay Thai fighter in traditional rope-wrapped shorts, hand wraps, a mongkhon headband, bare feet, and a muscular defined build",
     "masculine-presenting, focused, with Thai features"),
    ("a mushroom witch with a large red-and-white-speckled mushroom cap as a hat, a dress of overlapping fungi, bare gnarled feet, and spores floating around",
     "feminine-presenting, whimsical, with pale forest skin"),
    ("a Mongolian eagle hunter in a long embroidered chapan coat with fur trim, thick felt boots, a fox-fur hat, and a leather gauntlet with a golden eagle perched",
     "masculine-presenting, stoic, with Mongolian features"),
    ("a fire ant soldier standing upright with a segmented glossy red insect body, mandibles, six limbs (two as arms, four as legs), and tiny armor plates",
     "insectoid creature character, alien and martial"),
    ("a librarian ghost floating slightly above the ground in a long Victorian dress, translucent body, spectral book chains around the arms, and glowing eyes",
     "feminine-presenting, ethereal, pale translucent features"),
    ("a lumberjack in a red plaid flannel shirt, worn jeans, heavy work boots, leather suspenders, and carrying a large axe over one shoulder",
     "masculine-presenting, broad, with Canadian Scandinavian features"),
    ("a kitsune fox spirit in human form wearing a traditional miko outfit — white kosode, red hakama, tabi socks, zori sandals, with visible fox ears and three fluffy tails",
     "feminine-presenting, playful, with Japanese features"),
    ("a deep sea anglerfish person with a bulbous humanoid head, bioluminescent lure, sharp teeth, a dark diving suit, webbed hands, and flipper boots",
     "androgynous deep-sea creature, unsettling glow"),
    ("a vintage circus strongman in a leopard-print unitard, a curled waxed mustache, a wide leather belt, bare feet, and hoisting a huge barbell",
     "masculine-presenting, burly, with handlebar mustache"),
    ("a celestial being whose body is made of the night sky — deep blue translucent form with glowing star patterns, a nebula scarf, and a crescent moon crown",
     "androgynous cosmic being, radiant"),
    ("a Dutch Golden Age merchant in a black doublet with a large white ruff collar, puffed breeches, black stockings, buckled shoes, and a wide-brimmed hat",
     "masculine-presenting, dignified, with Dutch features"),
    ("a baby owl wizard with huge round eyes, a tiny feathered body, a miniature pointed wizard hat, a purple cape, and a tiny wand in one wing",
     "androgynous baby owl character, adorable"),
    ("a parkour runner in sleek athletic compression gear, lightweight trail runners, fingerless gloves, a small running pack, and a focused expression",
     "masculine-presenting, agile, with French Algerian features"),
    ("an Aztec jaguar warrior in a jaguar pelt costume with the jaguar head as a helmet, obsidian macuahuitl sword, round feathered shield, and wooden sandals",
     "masculine-presenting, fierce, with Mesoamerican features"),
    ("a sentient lamp post character with a thin wrought-iron body, a glowing bulb head, thin wire arms, a tiny door in the chest, and cobblestone base for feet",
     "androgynous object character, whimsical"),
    ("a K-pop idol in an oversized designer jacket, a crop top, leather pants, chunky platform sneakers, layered silver chains, and styled two-tone hair",
     "masculine-presenting, trendy, with Korean features"),
    ("a shadow assassin made of living darkness with a smooth featureless black body, glowing white eyes, wisps of shadow trailing, and twin daggers",
     "androgynous shadow creature, menacing"),
    ("a grandpa fisherman bear standing upright in a knitted fisherman's sweater, waders, a bucket hat, tall rubber boots, and holding a fishing rod",
     "masculine-presenting bear character, grandfatherly"),
    ("a solarpunk gardener in a living-plant-embedded vest of moss, vine-wrapped arms, recycled fabric pants, sandals made of bark, and a wide hat with a small solar panel",
     "non-binary-presenting, warm, with African features"),
    ("a heavy metal guitarist in a black spiked leather jacket, torn leather pants, spiked boots, studded wristbands, face paint, and a flying V guitar",
     "masculine-presenting, wild, with long black hair"),
    ("a Vietnamese quan ho singer in an ao tu than four-panel silk dress with a wide belt sash, a conical non la hat, and carrying a folding fan",
     "feminine-presenting, melodious, with Vietnamese features"),
    ("a carnival queen in an ornate feathered samba costume, jeweled bikini top and briefs, platform heels, elaborate headdress, and glittering body paint",
     "feminine-presenting, radiant, with Brazilian features"),
    ("a Scottish bagpiper in a traditional kilt, knee-length socks with red flashes, a sporran, ghillie brogues, a tartan sash, and holding bagpipes",
     "masculine-presenting, proud, with Scottish features"),
    ("a Maasai warrior in traditional red shuka cloth wrapped across the shoulder, leather sandals, beaded neck collars, carrying a long wooden spear",
     "masculine-presenting, tall and lean, with Kenyan Maasai features"),
    ("an Inuit hunter in a thick fur parka with fur-trimmed hood, sealskin boots, mittens, a carved bone knife in the belt, and a harpoon",
     "masculine-presenting, weathered, with Inuit features"),
    ("a flamenco dancer in a ruffled red polka-dot dress with a long train, black high-heeled shoes, a red rose in the hair, and castanets",
     "feminine-presenting, fiery, with Spanish features"),
    ("a Japanese yokai kappa standing upright — green reptilian skin, webbed hands and feet, a water-filled dish on the head, a simple loincloth, and carrying a fish",
     "androgynous yokai creature, mischievous"),
    ("a Tibetan monk in deep maroon and saffron robes, simple leather sandals, a shaved head, wooden prayer beads on the wrist, and hands in a mudra",
     "masculine-presenting, serene, with Tibetan features"),
    ("a West African griot musician in a flowing boubou with gold embroidery, a matching cap, leather sandals, and a kora held upright",
     "masculine-presenting, regal, with Senegalese features"),
    ("a roboticist inventor in a lab coat over a shirt and tie, safety goggles pushed up, rubber gloves, and holding a screwdriver next to a small floating drone",
     "non-binary-presenting, inventive, with South Asian features"),
    ("a punk rocker in a ripped band t-shirt, plaid miniskirt, torn fishnet stockings, chunky combat boots, spiked collar, and a tall green mohawk",
     "feminine-presenting, rebellious, with British features"),
    ("a firefighter in a yellow turnout coat, reinforced pants, helmet, thick gloves, steel-toed boots, and carrying an axe",
     "masculine-presenting, strong, with Irish American features"),
    ("a stage magician in a black tuxedo with tails, a top hat, a red-lined cape, white gloves, a wand, and a rabbit peeking from the hat",
     "masculine-presenting, suave, with handlebar mustache"),
    ("a Venetian carnival performer in a bauta mask, a tricorn hat, a long black cloak, fine breeches, white stockings, and buckled shoes",
     "androgynous, mysterious, masked features"),
    ("a coconut palm tree creature standing upright — a textured brown trunk body, green frond arms, coconut ornaments, root feet, and a cheerful face",
     "androgynous plant creature, sunny"),
    ("a cottagecore farmer's daughter in a linen puff-sleeve blouse, a long gingham skirt, a woven straw basket on the arm, lace-up ankle boots, and flower crown",
     "feminine-presenting, wholesome, with freckled pale skin"),
    ("a Maori warrior with a traditional piupiu flax skirt, bare chest with intricate tā moko tattoos, bone pendant, bare feet, and holding a carved taiaha staff",
     "masculine-presenting, powerful, with Maori features"),
    ("a retro-futurist astronaut in a chrome streamlined space suit with fishbowl helmet, a tall antenna, bulbous boots, and holding a ray gun",
     "masculine-presenting, 1950s scifi aesthetic, square jaw visible through visor"),
    ("a Russian ballet noble in a red velvet dress with gold trim, pearl necklace, heeled Mary Janes, elbow gloves, and a feathered fan",
     "feminine-presenting, regal, with Russian features"),
    ("a jazz saxophonist in a sharp pinstripe suit, a fedora tilted low, two-tone oxford shoes, suspenders visible under the jacket, and a tenor saxophone",
     "masculine-presenting, smooth, with Black features"),
    ("a naval admiral in a pristine white uniform with gold epaulettes, a peaked cap, medals on the chest, white gloves, polished black shoes",
     "masculine-presenting, commanding, with Eastern European features"),
]


@dataclass
class Prompt:
    id: str
    style: str
    character: str
    demographic: str
    pose: str
    pose_tag: str

    def full_text(self) -> str:
        finish = STYLE_FINISHES.get(self.style, "")
        return (
            f"{self.style} of {self.character}. "
            f"The character is {self.demographic}. "
            f"Posed {self.pose}. "
            f"{finish} "
            f"Single character only, centered in the frame, full body visible. "
            f"Solid plain white background. No text, no watermark, no UI elements."
        )

    def as_markdown_entry(self) -> str:
        return f"{self.id}. {self.full_text()}"


def generate_prompts(count: int, start_id: int, seed: int) -> list[Prompt]:
    rng = random.Random(seed)

    # Rotate through combinations to guarantee diversity
    combos: list[tuple[str, str, str, str, str]] = []
    for i in range(count):
        style = STYLES[i % len(STYLES)]
        char, demo = CHARACTERS[i % len(CHARACTERS)]
        pose, pose_tag = POSES[i % len(POSES)]
        combos.append((style, char, demo, pose, pose_tag))

    # Shuffle slightly so we don't get predictable repetitions of pose=neutral
    # for every Nth entry (style, character, pose all cycle at different rates
    # so we already get variety; leave order as-is for reproducibility).

    prompts = []
    for i, (style, char, demo, pose, pose_tag) in enumerate(combos):
        pid = f"SC-{start_id + i:03d}"
        prompts.append(Prompt(
            id=pid, style=style, character=char,
            demographic=demo, pose=pose, pose_tag=pose_tag,
        ))

    return prompts


def main():
    parser = argparse.ArgumentParser(
        description="Generate diverse self-contained character prompts."
    )
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--start_id", type=int, default=61,
                        help="First SC-### ID to use")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None,
                        help="Write to file; otherwise print to stdout")
    args = parser.parse_args()

    prompts = generate_prompts(args.count, args.start_id, args.seed)

    lines = []
    for p in prompts:
        lines.append(p.as_markdown_entry())
        lines.append("")

    text = "\n".join(lines)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"Wrote {len(prompts)} prompts to {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
