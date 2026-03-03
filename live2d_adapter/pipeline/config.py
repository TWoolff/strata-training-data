"""Minimal config for the standalone Live2D adapter.

Contains only the constants needed by live2d_renderer.py, live2d_mapper.py,
live2d_review_ui.py, and moc3_parser.py.
"""

from __future__ import annotations

RegionId = int
RGB = tuple[int, int, int]

REGION_NAMES: dict[RegionId, str] = {
    0: "background",
    1: "head",
    2: "neck",
    3: "chest",
    4: "spine",
    5: "hips",
    6: "shoulder_l",
    7: "upper_arm_l",
    8: "forearm_l",
    9: "hand_l",
    10: "shoulder_r",
    11: "upper_arm_r",
    12: "forearm_r",
    13: "hand_r",
    14: "upper_leg_l",
    15: "lower_leg_l",
    16: "foot_l",
    17: "upper_leg_r",
    18: "lower_leg_r",
    19: "foot_r",
    20: "accessory",
    21: "hair_back",
}

# Number of joint regions (body regions only, excluding background)
NUM_JOINT_REGIONS: int = 19  # regions 1–19 (core body; accessory/hair_back handled separately)

# Bounding box padding for joint extraction (fraction of bbox dimension)
JOINT_BBOX_PADDING: float = 0.05

# Render settings
# ---------------------------------------------------------------------------

RENDER_RESOLUTION: int = 512
CAMERA_TYPE: str = "ORTHO"

# Left↔right region ID swap pairs for flip augmentation
FLIP_REGION_SWAP: dict[RegionId, RegionId] = {
    6: 10,  # shoulder_l ↔ shoulder_r
    7: 11,  # upper_arm_l ↔ upper_arm_r
    8: 12,  # forearm_l ↔ forearm_r
    9: 13,  # hand_l ↔ hand_r
    10: 6,
    11: 7,
    12: 8,
    13: 9,
    14: 17,  # upper_leg_l ↔ upper_leg_r
    15: 18,  # lower_leg_l ↔ lower_leg_r
    16: 19,  # foot_l ↔ foot_r
    17: 14,
    18: 15,
    19: 16,
}

# Left↔right joint name swap pairs for flip augmentation

# ---------------------------------------------------------------------------

REGION_NAME_TO_ID: dict[str, RegionId] = {name: rid for rid, name in REGION_NAMES.items()}

# ---------------------------------------------------------------------------
# Live2D fragment-to-Strata label mapping
# ---------------------------------------------------------------------------
# Ordered list of (regex_pattern, strata_region_name) tuples.
# Patterns are matched case-insensitively against Live2D ArtMesh fragment names.
# First match wins — place specific patterns before general ones.
# Covers English, Japanese romaji, and common Live2D naming conventions.

# Patterns are matched case-insensitively against Live2D ArtMesh fragment names.
# First match wins — place specific patterns before general ones.
# Covers English, Japanese romaji, and common Live2D naming conventions.

LIVE2D_FRAGMENT_PATTERNS: list[tuple[str, str]] = [
    # =========================================================================
    # EXPLICIT EXCLUSIONS — non-body content that would otherwise false-match
    # Must come before body patterns.
    # =========================================================================
    # Effects, particles, backgrounds, UI elements
    (
        r"^(yanhua|yanwu|guangqiu|yinying|shadow|effect|light|spark|confetti"
        r"|xuehua|meigui|jizhongxian|suduxian|bianpao|guangxian|fanshe|fansheguang"
        r"|dengguang|shangguang|bg|background|stage|scene|prop|item|weapon|shield"
        r"|daoying|dimian|shitou|lanhua|zhiwu|floor|wall|ground|water|cloud|sky"
        r"|star|circle|ring|line|dot|spot|flash|glow|bloom|aura|halo|particle"
        r"|bubble|smoke|fire|flame|wind|ribbon|flower|petal|leaf|grass|tree"
        r"|chair|sofa|table|bed|pillow|book|bag|basket|bottle|cup|box|cage"
        r"|gun|sword|staff|rod|wand|axe|spear|bow|arrow|instrument|mic"
        r"|coin|gem|jewel|crystal|stone|rock|sand|ice|snow|rain|wave"
        r"|logo|text|ui|button|frame|border|panel|icon|mark|symbol"
        r"|(cloth|clothing|costume|outfit|garment|dress|skirt|pants|trousers|shorts)"
        r"|(hat|cap|helmet|crown|headwear|headgear|hood)"
        r"|(cape|cloak|coat|jacket|vest|shirt|blouse|top|bottom|bra|underwear)"
        r"|(armor|armour|pauldron|gauntlet|greave|shield|weapon)"
        r"|(accessory|acc|ornament|pendant|brooch|badge|pin|decoration)"
        r"|^[a-z]{1,2}[0-9]+$|^artmesh[0-9]+$)",
        "background",
    ),

    # =========================================================================
    # HAIR — very common, before head to avoid "head" swallowing hair meshes
    # =========================================================================
    # Hair back (most specific first)
    (r"hair.*back|back.*hair|hairback|hair_b[^r]|houfa|houfal|houfar|mawei|ponytail"
     r"|horsetail|twintail|braid|pigtail|ahoge|odango|ushirogami", "hair_back"),
    # Hair side
    (r"hair.*side|side.*hair|hairside|sidehair|mimi.*hair|templelock", "hair_back"),
    # Hair front (bangs/fringe)
    (r"hair.*front|front.*hair|hairfront|liuhai|bangs|fringe|maegami|ahoge_f"
     r"|forehair|前发|刘海", "head"),
    # General hair
    (r"hair|toufa|kami[^n]|wig|bun|chignon|tress|strand|lock", "head"),

    # =========================================================================
    # HEAD — face and facial features
    # =========================================================================
    # Eyes (most common — very specific patterns to avoid false positives)
    (r"eye[_\s]|_eye|eyelash|eyeball|eyelid|eyebrow|^eye$|^eye[lr]$|eye[lr][_\d]"
     r"|me_[lr]|jiemao|meimao"
     r"|yanqiu|yanjing|tongkong|yanren|mabuta|hitomi|pupil|iris|sclera"
     r"|shang.*yan$|xia.*yan$|yan$|^yan[0-9]|yanzhu|yanbai|yanjiao|yanxian"
     r"|yanying_y|yanzhuGG|眼睛|五官", "head"),
    # Eyebrow
    (r"brow|mayu|meimao|mei[lr]$|^meil$|^meir$", "head"),
    # Nose
    (r"nose|bizi|hana(?!k)|biyan|鼻", "head"),
    # Mouth/lips/teeth
    (r"mouth|lip[^s]|lips|tongue|teeth|zui|zuiba|chun|kuchi|\bha\b|yachi|齿|唇|嘴", "head"),
    # Ears
    (r"\bear\b|ear[_\s]|_ear|ears|erduo|mimi(?!k)|jier|耳", "head"),
    # Blush/cheek
    (r"blush|cheek|hongzui|hongkuai|jiahong|face.*color|lian_hong|腮红", "head"),
    # Face/head general
    (r"face|lian(?!j)|^lian\d*$|kao(?!s)|^kao\d*$|顔|脸|臉", "head"),
    (r"\bhead\b|^head_|_head$|atama|tou(?![fp])|^tou\d*$|naodai|noggin|skull|cranium"
     r"|颅|头部", "head"),

    # =========================================================================
    # NECK
    # =========================================================================
    # Exclude necktie/necklace/neckband/neckribbon → background
    (r"necktie|necklace|neckband|neckribbon|neckbow|neckline|领带|领结|项链", "background"),
    # bozi + decorative suffix → background (butterfly bow, ribbon, etc.)
    (r"bozi(?:hudiejie|hudie|dai|ribbon|bow|tie|knot|jie)", "background"),
    # Real neck
    (r"\bneck\b|neck[_\s\d]|_neck|^bozi\d*$|rebozi|jingbu|kubi(?!r)|nodo|喉|頸|脖|颈", "neck"),

    # =========================================================================
    # SHOULDERS — lateralized first (specific before general)
    # =========================================================================
    # Left shoulder: zuo+jian, shoulder_l, shoulder_left, jianbang_l
    (
        r"shoulder.*_?l\b|shoulder.*left|left.*shoulder"
        r"|zuo.*jian\b|jian.*zuo|jianbang.*zuo|zuojianbang"
        r"|kata.*l\b|kata.*hidari|jian_l|jianbang_l|肩.*左|左.*肩",
        "shoulder_l",
    ),
    # Right shoulder
    (
        r"shoulder.*_?r\b|shoulder.*right|right.*shoulder"
        r"|you.*jian\b|jian.*you|jianbang.*you|youjianbang"
        r"|kata.*r\b|kata.*migi|jian_r|jianbang_r|肩.*右|右.*肩",
        "shoulder_r",
    ),
    # Unlateralized shoulder → chest (fallback)
    (r"\bshoulder\b|jianbang|kata[^k]|肩", "chest"),

    # =========================================================================
    # ARMS — forearm before upper_arm (forearm is substring of upper_arm contexts)
    # =========================================================================
    # LEFT FOREARM: qianbi_l, forearm_l, arm_fore_l, xiaoshoubi_l, xiabei_l
    (
        r"forearm.*_?l\b|forearm.*left|left.*forearm"
        r"|arm.*(?:fore|lower|xia|small).*_?l\b|arm.*lower.*left"
        r"|zuo.*(?:qianbi|shoubi|xiabei|xiaoshoubi)|(?:qianbi|shoubi|xiabei).*zuo"
        r"|zuoshoubi|shoubizuo|armfore.*l|arm_l_(?:fore|under|lower)"
        r"|前臂.*左|左.*前臂|小臂.*左|左.*小臂",
        "forearm_l",
    ),
    # LEFT UPPER ARM: shangbi_l, arm_upper_l, dashoubi_l, ude_ue_L, arm_upper_left
    (
        r"upper.*arm.*_?l\b|arm.*upper.*_?l\b|upper.*arm.*left"
        r"|zuo.*(?:shangbi|dashoubi)|(?:shangbi|dashoubi).*zuo"
        r"|zuoshangbi|shangbizuo|arm.*shang.*_?l\b|shangbi_l"
        r"|ude_ue[_\s]*l|arm_upper_left|upperarm.*l\b|arm.*upper.*left"
        r"|大臂.*左|左.*大臂|上臂.*左|左.*上臂",
        "upper_arm_l",
    ),
    # LEFT FOREARM generic variants: ArmALFore, ArmLFore
    (r"arm[a-z]*.*l.*fore|arm.*fore.*l\b|arm[a-z]*l[_\s]*fore", "forearm_l"),
    # LEFT ARM generic (after upper/fore to avoid swallowing them)
    (
        r"arm_l\b|arm_l[_\d]|arm_left|left.*arm\b|_arm_l[_\s\d]|arml\b"
        r"|^arm[_\d]*_?l[_\d]*$"
        r"|zuo.*bi\b|bi.*zuo|shoubi_l|arm.*_l\b(?!.*(?:upper|fore|shang|qian))"
        r"|ude_l|ude_hidari",
        "upper_arm_l",
    ),
    # RIGHT FOREARM
    (
        r"forearm.*_?r\b|forearm.*right|right.*forearm"
        r"|arm.*(?:fore|lower|xia|small).*_?r\b|arm.*lower.*right"
        r"|you.*(?:qianbi|shoubi|xiabei|xiaoshoubi)|(?:qianbi|shoubi|xiabei).*you"
        r"|youshoubi|shoubiyou|armfore.*r|arm_r_(?:fore|under|lower)"
        r"|前臂.*右|右.*前臂|小臂.*右|右.*小臂",
        "forearm_r",
    ),
    # RIGHT UPPER ARM: shangbi_r, arm_upper_r, dashoubi_r, ude_ue_R, arm_upper_right
    (
        r"upper.*arm.*_?r\b|arm.*upper.*_?r\b|upper.*arm.*right"
        r"|you.*(?:shangbi|dashoubi)|(?:shangbi|dashoubi).*you"
        r"|youshangbi|shangbiyou|arm.*shang.*_?r\b|shangbi_r"
        r"|ude_ue[_\s]*r|arm_upper_right|upperarm.*r\b|arm.*upper.*right"
        r"|大臂.*右|右.*大臂|上臂.*右|右.*上臂",
        "upper_arm_r",
    ),
    # RIGHT FOREARM generic variants: ArmARFore, ArmRFore
    (r"arm[a-z]*.*r.*fore|arm.*fore.*r\b|arm[a-z]*r[_\s]*fore", "forearm_r"),
    # RIGHT ARM generic
    (
        r"arm_r\b|arm_r[_\d]|arm_right|right.*arm\b|_arm_r[_\s\d]|armr\b"
        r"|^arm[_\d]*_?r[_\d]*$"
        r"|you.*bi\b|bi.*you|shoubi_r|arm.*_r\b(?!.*(?:upper|fore|shang|qian))"
        r"|ude_r|ude_migi",
        "upper_arm_r",
    ),

    # =========================================================================
    # HANDS — after arms, lateralized first
    # =========================================================================
    # Left hand: zuoshou, hand_l, shou_l
    (
        r"hand.*_?l\b|hand.*left|left.*hand"
        r"|zuoshou|shou.*zuo|te_l|te_hidari|hand_l"
        r"|wrist.*l|palm.*l|knuckle.*l|finger.*l"
        r"|手.*左|左.*手",
        "hand_l",
    ),
    # Right hand: youshou, hand_r, shou_r
    (
        r"hand.*_?r\b|hand.*right|right.*hand"
        r"|youshou|shou.*you|te_r|te_migi|hand_r"
        r"|wrist.*r|palm.*r|knuckle.*r|finger.*r"
        r"|手.*右|右.*手",
        "hand_r",
    ),
    # Generic hand / fingers / palm (unlateralized → chest fallback)
    # Note: shouzhang/shizhi must come BEFORE head patterns to prevent false head match
    (
        r"\bhand\b|\bshou\b|shouzhang|shizhi|wumingzhi|wumingzi\b"
        r"|zhongjian|xiaozhijian|muzhijian|shizhijian|zhongjiao\b"
        r"|finger|thumb|palm|wrist|knuckle|fist"
        r"|yubi|oyayubi|hitosashi|nakayubi|kusuri|koyubi|te\b",
        "chest",
    ),

    # =========================================================================
    # LEGS — lower_leg before upper_leg (xiaotui before tudui/tui)
    # =========================================================================
    # LEFT LOWER LEG: xiaotui_l, shin_l, calf_l, leg_lower_l
    (
        r"lower.*leg.*_?l\b|leg.*lower.*_?l\b|lower.*leg.*left"
        r"|shin.*_?l\b|shin.*left|calf.*_?l\b|calf.*left"
        r"|zuo.*xiaotui|xiaotui.*zuo|zuoxiaotui|xiaotui_l"
        r"|sune.*l|ankle.*l|小腿.*左|左.*小腿",
        "lower_leg_l",
    ),
    # LEFT UPPER LEG: tudui_l, thigh_l, leg_upper_l
    (
        r"upper.*leg.*_?l\b|leg.*upper.*_?l\b|upper.*leg.*left"
        r"|thigh.*_?l\b|thigh.*left|momo.*_?l\b|momo.*left"
        r"|zuo.*(?:tudui|datui)|(?:tudui|datui).*zuo|zuotudui|tudui_l"
        r"|大腿.*左|左.*大腿",
        "upper_leg_l",
    ),
    # LEFT LEG generic
    (
        r"leg.*_?l\b|leg.*left|left.*leg\b|_leg_l[_\s]|legl\b"
        r"|zuo.*tui\b|tui.*zuo|zuotui|leg_l\b"
        r"|腿.*左|左.*腿",
        "upper_leg_l",
    ),
    # RIGHT LOWER LEG
    (
        r"lower.*leg.*_?r\b|leg.*lower.*_?r\b|lower.*leg.*right"
        r"|shin.*_?r\b|shin.*right|calf.*_?r\b|calf.*right"
        r"|you.*xiaotui|xiaotui.*you|youxiaotui|xiaotui_r"
        r"|sune.*r|ankle.*r|小腿.*右|右.*小腿",
        "lower_leg_r",
    ),
    # RIGHT UPPER LEG
    (
        r"upper.*leg.*_?r\b|leg.*upper.*_?r\b|upper.*leg.*right"
        r"|thigh.*_?r\b|thigh.*right|momo.*_?r\b|momo.*right"
        r"|you.*(?:tudui|datui)|(?:tudui|datui).*you|youtudui|tudui_r"
        r"|大腿.*右|右.*大腿",
        "upper_leg_r",
    ),
    # RIGHT LEG generic
    (
        r"leg.*_?r\b|leg.*right|right.*leg\b|_leg_r[_\s]|legr\b"
        r"|you.*tui\b|tui.*you|youtui|leg_r\b"
        r"|腿.*右|右.*腿",
        "upper_leg_r",
    ),

    # =========================================================================
    # FEET — lateralized first
    # =========================================================================
    # Left foot: zuojiao, foot_l, ashi_l, l_jiao, jiaozhil
    (
        r"foot.*_?l[\d_]?|foot.*left|left.*foot|foot_l\b"
        r"|zuojiao|jiao.*zuo|jiaozuo|l_jiao\d*|jiao_l|jiaozhil|jiaol\b"
        r"|ashi.*l|ashi.*hidari|zu_l"
        r"|toe.*l|heel.*l|ankle.*l\b|脚.*左|左.*脚|足.*左|左.*足",
        "foot_l",
    ),
    # Right foot: youjiao, foot_r, ashi_r, r_jiao, jiaozhi_r
    (
        r"foot.*_?r[\d_]?|foot.*right|right.*foot|foot_r\b"
        r"|youjiao|jiao.*you|jiaoyou|r_jiao\d*|jiao_r|jiaozhir|jiaor\b"
        r"|ashi.*r|ashi.*migi|zu_r"
        r"|toe.*r|heel.*r|ankle.*r\b|脚.*右|右.*脚|足.*右|右.*足",
        "foot_r",
    ),
    # Generic foot/ankle/toe numbered (foot2, foot3, etc.) → hips fallback
    (r"^foot\d+$|^jiao\d+$|\bfoot\b|\bfeet\b|\btoe\b|\bheel\b|\bankle\b"
     r"|jiao\b|ashi\b|靴|足\b|脚\b", "hips"),
    # Generic leg (no lateralization) → hips fallback
    (r"^\bleg\b$|^leg\d+$|\bleg\b", "hips"),
    # Generic arm (no lateralization, e.g. "ArmA", "ArmB", bare "arm") → chest fallback
    (r"^\barm[ab]$|^arm[ab]\d+$|\barm\b", "chest"),
    # Generic hand numbered (hand1, hand61, etc.) → chest fallback
    (r"\bhand\d+$|\bhand[a-z]+\d+$", "chest"),

    # =========================================================================
    # HIPS / WAIST / PELVIS
    # =========================================================================
    (
        r"\bhip\b|hips|pelvis|koshi|waist|yao[_\d]|^yao\d*$|yaohou|yaoqian"
        r"|yaodai|yaoshili|yaoshi|yaolan"
        r"|tun\b|kua[_\d]|^kua\d*$"
        r"|lower.*body|bottom|buttock|glute|crotch|groin|abdomen|belly"
        r"|fubu|duzi|腰|臀|胯|裆|下半身|腹|肚",
        "hips",
    ),
    # datui (大腿 = thigh) — lateralized and numbered variants
    (r"datui.*[lr]\b|[lr].*datui|\bdatui\d*$|datui_", "hips"),
    # Generic tui (leg) — r_tui, l_tui, qunxiaotui etc.
    (r"^[rl]_tui\d*$|^tui[lr]\d*$|qun.*tui|tui_[lr]", "hips"),

    # =========================================================================
    # CHEST / TORSO
    # =========================================================================
    (
        r"chest[\d_]|^chest\d*$|breast|\bbust\b|bust[_\d]|pectoral|xiong[\d_]|^xiong\d*$"
        r"|fuxiong|zuoxiong|youxiong|xiongbu|mune\b|bosom|oppai"
        r"|torso|trunk|上半身|胸",
        "chest",
    ),
    (r"\bchest\b|\bxiong\b", "chest"),
    (r"\bbody\b|karada|upperbody|upper.*body|躯干|身体|上身", "chest"),

    # =========================================================================
    # SPINE / BACK
    # =========================================================================
    (r"spine|senaka|back(?!ground|hair|bone|pack|drop|ground)|後ろ|背中|背部|脊", "spine"),

    # =========================================================================
    # CHINESE unicode body part characters (catch-all for Chinese-labeled models)
    # =========================================================================
    (r"眼|瞳|眉|睫|嘴|唇|鼻|耳|发|刘海|脸|颊|面|头|髪|前髪", "head"),
    (r"脖|颈|頸", "neck"),
    (r"肩.*左|左.*肩", "shoulder_l"),
    (r"肩.*右|右.*肩", "shoulder_r"),
    (r"前臂.*左|左.*前臂|小臂.*左|左.*小臂", "forearm_l"),
    (r"大臂.*左|左.*大臂|上臂.*左|左.*上臂", "upper_arm_l"),
    (r"手.*左|左.*手", "hand_l"),
    (r"前臂.*右|右.*前臂|小臂.*右|右.*小臂", "forearm_r"),
    (r"大臂.*右|右.*大臂|上臂.*右|右.*上臂", "upper_arm_r"),
    (r"手.*右|右.*手", "hand_r"),
    (r"小腿.*左|左.*小腿", "lower_leg_l"),
    (r"大腿.*左|左.*大腿", "upper_leg_l"),
    (r"脚.*左|左.*脚|足.*左|左.*足", "foot_l"),
    (r"小腿.*右|右.*小腿", "lower_leg_r"),
    (r"大腿.*右|右.*大腿", "upper_leg_r"),
    (r"脚.*右|右.*脚|足.*右|右.*足", "foot_r"),
    (r"臀|裆|胯", "hips"),
    (r"胸|上身|躯干", "chest"),
    (r"肩\b", "chest"),
    (r"臂|腕", "chest"),
    # Chinese CDI Part name fragments (whole-word Chinese body terms)
    (r"^腿$", "hips"),
    (r"^大臂$", "upper_arm_l"),   # ambiguous side but maps to some arm
    (r"^小臂$", "forearm_l"),     # ambiguous side
    (r"^手$", "chest"),           # bare "hand" → chest (unlateralized)
    (r"^身体$", "chest"),
    (r"^脸$", "head"),
    (r"^眼睛$", "head"),
    (r"^五官$", "head"),
]

# ---------------------------------------------------------------------------
# Spine bone/slot-to-Strata label mapping
# ---------------------------------------------------------------------------
# Ordered list of (regex_pattern, strata_region_name) tuples.
# Patterns are matched case-insensitively against Spine bone or slot names.
# First match wins — place specific patterns before general ones.
# Covers common Spine naming conventions (hyphenated and underscore variants).

# Live2D renderer augmentation settings
# ---------------------------------------------------------------------------
# Rotation angles (degrees) applied to Live2D composites for data augmentation.
LIVE2D_AUGMENTATION_ROTATIONS: list[float] = [-5.0, 0.0, 5.0]

# Uniform scale factors applied to Live2D composites for data augmentation.
LIVE2D_AUGMENTATION_SCALES: list[float] = [0.9, 1.0, 1.1]

# Color jitter ranges: (hue_shift_degrees, saturation_factor, brightness_factor).
# Each element is (min, max) for random uniform sampling.
LIVE2D_AUGMENTATION_COLOR_JITTER: dict[str, tuple[float, float]] = {
    "hue": (-15.0, 15.0),  # degrees on hue wheel
    "saturation": (0.85, 1.15),  # multiplicative factor
    "brightness": (0.85, 1.15),  # multiplicative factor
}

# ---------------------------------------------------------------------------
# Contour rendering (Freestyle line pairs for contour-removal training)
# ---------------------------------------------------------------------------

# Freestyle edge detection flags — which edge types to render as contour lines.

