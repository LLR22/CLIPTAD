#-------------------------------标签对应的文本属性及属性分解字典------------------------------#
# Original action_to_id mapping
action_to_id = {
    '伸懒腰': 0, '倒水': 1, '写字': 2, '切水果': 3, '吃水果': 4, '吃药': 5,
    '喝水': 6, '坐下': 7, '开关护眼灯': 8, '开关窗帘': 9, '开关窗户': 10, '打字': 11,
    '打开信封': 12, '扔垃圾': 13, '拿水果': 14, '捡东西': 15, '接电话': 16,
    '操作鼠标': 17, '擦桌子': 18, '板书': 19, '洗手': 20, '玩手机': 21,
    '看书': 22, '给植物浇水': 23, '走向床': 24, '走向椅子': 25, '走向橱柜': 26,
    '走向窗户': 27, '走向黑板': 28, '起床': 29, '起立': 30, '躺下': 31,
    '静止站立': 32, '静止躺着': 33
}

# Translation mapping
translations = {
    '伸懒腰': 'Stretching', '倒水': 'Pouring Water', '写字': 'Writing', '切水果': 'Cutting Fruit',
    '吃水果': 'Eating Fruit', '吃药': 'Taking Medicine', '喝水': 'Drinking Water', '坐下': 'Sitting Down',
    '开关护眼灯': 'Turning On/Off Eye Protection Lamp', '开关窗帘': 'Opening/Closing Curtains',
    '开关窗户': 'Opening/Closing Windows', '打字': 'Typing', '打开信封': 'Opening Envelope',
    '扔垃圾': 'Throwing Garbage', '拿水果': 'Picking Fruit', '捡东西': 'Picking Up Items', '接电话': 'Answering Phone',
    '操作鼠标': 'Using Mouse', '擦桌子': 'Wiping Table', '板书': 'Writing on Blackboard', '洗手': 'Washing Hands',
    '玩手机': 'Using Phone', '看书': 'Reading', '给植物浇水': 'Watering Plants', '走向床': 'Walking to Bed',
    '走向椅子': 'Walking to Chair', '走向橱柜': 'Walking to Cabinet', '走向窗户': 'Walking to Window',
    '走向黑板': 'Walking to Blackboard', '起床': 'Getting Out of Bed', '起立': 'Standing Up',
    '躺下': 'Lying Down', '静止站立': 'Standing Still', '静止躺着': 'Lying Still', '走路': 'Walking'
}

# attribute decomposition
# action_attribute = {v: 'pad' for v in translations.values()}

action_attribute = {
    'Stretching': {
        'hand_action': 'Raising arms upwards, extending fingers',
        'torso_action': 'Slightly arching back',
        'leg_action': 'Standing with legs apart for balance'
    },
    'Pouring Water': {
        'hand_action': 'Gripping cup/pot, tilting wrist',
        'torso_action': 'Leaning forward slightly',
        'leg_action': 'Stationary stance'
    },
    'Writing': {
        'hand_action': 'Holding pen, making precise finger movements',
        'torso_action': 'Bending forward at desk',
        'leg_action': 'Seated or standing still'
    },
    'Cutting Fruit': {
        'hand_action': 'Holding knife with dominant hand, stabilizing fruit with other hand',
        'torso_action': 'Leaning over cutting board',
        'leg_action': 'Standing in place'
    },
    'Eating Fruit': {
        'hand_action': 'Bringing food to mouth, chewing motion',
        'torso_action': 'Upright sitting position',
        'leg_action': 'Legs relaxed under table'
    },
    'Taking Medicine': {
        'hand_action': 'Picking up pills, opening medicine bottle',
        'torso_action': 'Slight forward lean',
        'leg_action': 'Standing near medicine cabinet'
    },
    'Drinking Water': {
        'hand_action': 'Lifting cup to lips, swallowing',
        'torso_action': 'Head tilted back',
        'leg_action': 'Stationary standing/sitting'
    },
    'Sitting Down': {
        'hand_action': 'Pushing on armrests/desk',
        'torso_action': 'Lowering body',
        'leg_action': 'Bending knees, adjusting position'
    },
    'Turning On/Off Eye Protection Lamp': {
        'hand_action': 'Pressing switch/button',
        'torso_action': 'Facing the lamp',
        'leg_action': 'Standing within reach'
    },
    'Opening/Closing Curtains': {
        'hand_action': 'Grasping curtain fabric or cord',
        'torso_action': 'Sideways arm extension',
        'leg_action': 'Possible step stool use'
    },
    'Opening/Closing Windows': {
        'hand_action': 'Turning handle, pushing/pulling window',
        'torso_action': 'Leaning toward window',
        'leg_action': 'Weight shift for leverage'
    },
    'Typing': {
        'hand_action': 'Finger key pressing',
        'torso_action': 'Upright posture',
        'leg_action': 'Seated position'
    },
    'Opening Envelope': {
        'hand_action': 'Sliding finger along seal',
        'torso_action': 'Holding envelope at chest height',
        'leg_action': 'Stationary standing'
    },
    'Throwing Garbage': {
        'hand_action': 'Grasping trash, swinging arm toward bin',
        'torso_action': 'Twisting at waist',
        'leg_action': 'Stepping toward bin'
    },
    'Picking Fruit': {
        'hand_action': 'Plucking motion, hand-eye coordination',
        'torso_action': 'Reaching upward',
        'leg_action': 'Tip-toeing if needed'
    },
    'Picking Up Items': {
        'hand_action': 'Bending fingers to grasp',
        'torso_action': 'Squatting or bending at waist',
        'leg_action': 'Knee flexion for lower items'
    },
    'Answering Phone': {
        'hand_action': 'Holding phone to ear',
        'torso_action': 'Upright posture',
        'leg_action': 'Possibly pacing'
    },
    'Using Mouse': {
        'hand_action': 'Wrist movement, button clicking',
        'torso_action': 'Forward shoulder position',
        'leg_action': 'Seated legs stationary'
    },
    'Wiping Table': {
        'hand_action': 'Circular scrubbing motion',
        'torso_action': 'Leaning over table',
        'leg_action': 'Side-stepping along table'
    },
    'Writing on Blackboard': {
        'hand_action': 'Chalk grip, arm elevation',
        'torso_action': 'Facing board',
        'leg_action': 'Standing with weight shifts'
    },
    'Washing Hands': {
        'hand_action': 'Rubbing palms, finger interlacing',
        'torso_action': 'Bent over sink',
        'leg_action': 'Standing at basin'
    },
    'Using Phone': {
        'hand_action': 'Screen tapping/swiping',
        'torso_action': 'Forward neck flexion',
        'leg_action': 'Stationary or pacing'
    },
    'Reading': {
        'hand_action': 'Page turning, holding book',
        'torso_action': 'Relaxed sitting posture',
        'leg_action': 'Crossed legs possible'
    },
    'Watering Plants': {
        'hand_action': 'Squeezing trigger, aiming spout',
        'torso_action': 'Arm elevation',
        'leg_action': 'Walking between plants'
    },
    'Walking to Bed': {
        'hand_action': 'Possibly pulling back covers',
        'torso_action': 'Forward momentum',
        'leg_action': 'Alternating strides'
    },
    'Walking to Chair': {
        'hand_action': 'Arm swinging for balance',
        'torso_action': 'Upright posture',
        'leg_action': 'Controlled steps'
    },
    'Walking to Cabinet': {
        'hand_action': 'Possible door opening',
        'torso_action': 'Orientation toward target',
        'leg_action': 'Directional walking'
    },
    'Walking to Window': {
        'hand_action': 'Arm swing coordination',
        'torso_action': 'Facing forward',
        'leg_action': 'Steady gait'
    },
    'Walking to Blackboard': {
        'hand_action': 'Carrying chalk/marker',
        'torso_action': 'Forward lean during motion',
        'leg_action': 'Accelerating steps'
    },
    'Getting Out of Bed': {
        'hand_action': 'Pushing up on mattress',
        'torso_action': 'Core engagement for sitting up',
        'leg_action': 'Swinging legs over edge'
    },
    'Standing Up': {
        'hand_action': 'Pushing on knees/armrests',
        'torso_action': 'Vertical elevation',
        'leg_action': 'Knee extension'
    },
    'Lying Down': {
        'hand_action': 'Guiding body lowering',
        'torso_action': 'Controlled recline',
        'leg_action': 'Sequential leg lifting'
    },
    'Standing Still': {
        'hand_action': 'Relaxed at sides',
        'torso_action': 'Neutral spinal alignment',
        'leg_action': 'Even weight distribution'
    },
    'Lying Still': {
        'hand_action': 'Resting on surface',
        'torso_action': 'Full body contact with bed',
        'leg_action': 'Extended straight position'
    },
    'Walking': {
        'hand_action': 'Natural arm swing',
        'torso_action': 'Slight forward lean',
        'leg_action': 'Alternating leg propulsion'
    }
}

#------------------------建立映射--------------------------------#
# Create id_to_action mapping
id_to_action = {str(v): translations[k] for k, v in action_to_id.items()}

english_action_to_id = {translations[k]: v for k, v in action_to_id.items()}
# Output results
# print("action_to_id:", action_to_id)
# print("id_to_action:", id_to_action)

'''
将走路合并
'''
# 创建新的标签映射（30类）
old_to_new_mapping = {}
new_action_to_id = {}
current_id = 0
# 合并的“走路”相关动作
walk_actions = ['走向床', '走向椅子', '走向橱柜', '走向窗户', '走向黑板']

for action, old_id in action_to_id.items():
    if action in walk_actions:
        if '走路' not in new_action_to_id:  # 将所有走路相关动作合并为一类
            new_action_to_id['走路'] = current_id
            current_id += 1
        old_to_new_mapping[old_id] = new_action_to_id['走路']
    else:
        if action not in new_action_to_id:
            new_action_to_id[action] = current_id
            current_id += 1
        old_to_new_mapping[old_id] = new_action_to_id[action]

new_id_to_action = {v: translations[k] for k, v in new_action_to_id.items()}

# 属性分解后的映射
id_to_attribute = {k: action_attribute[v] for k, v in new_id_to_action.items()}
# 输出新映射
# print("Old to New Mapping:", old_to_new_mapping)
# print("New Action to ID:", new_action_to_id)
# print("new_id_to_action", new_id_to_action)


#------------------分解动作的开始和结束阶段---------------#
action_breakdown_start = {
    # 手部初始动作 | 躯干准备姿势 | 腿部启动机制
    'Stretching': {
        'hand_start': 'Arms initiate upward trajectory from resting position',
        'torso_start': 'Intercostal muscles engage for ribcage expansion',
        'leg_start': 'Calcaneal contact area increases for base of support'
    },
    'Pouring Water': {
        'hand_start': 'Thenar eminence contacts container handle',
        'torso_start': 'Sternocleidomastoid activates for head stabilization',
        'leg_start': 'Subtalar joint pronation for weight shift'
    },
    'Writing': {
        'hand_start': 'Tripod grasp formation on writing instrument',
        'torso_start': 'Thoracic kyphosis increases for desk proximity',
        'leg_start': 'Ischial tuberosity pressure distribution (if seated)'
    },
    'Cutting Fruit': {
        'hand_start': 'Force closure between knife handle and palm',
        'torso_start': 'Abdominal bracing for core stability',
        'leg_start': 'Bilateral hip external rotation for stance width'
    },
    'Eating Fruit': {
        'hand_start': 'Prehensile adjustment of food grip',
        'torso_start': 'Cervical flexion initiates head lowering',
        'leg_start': 'Plantar pressure shifts to metatarsal heads'
    },
    'Taking Medicine': {
        'hand_start': 'Pincer grasp on pill/capsule',
        'torso_start': 'Sternal elevation for respiratory preparation',
        'leg_start': 'Anterior tibialis activation for balance'
    },
    'Drinking Water': {
        'hand_start': 'Thenar-hypothenar compression on cup',
        'torso_start': 'Hyoid bone elevation initiates swallowing prep',
        'leg_start': 'Gastrocnemius quiet standing activation'
    },
    'Sitting Down': {
        'hand_start': 'Palmar contact on support surface',
        'torso_start': 'Pelvic posterior tilt initiates descent',
        'leg_start': 'Eccentric quadriceps loading begins'
    },
    'Turning On/Off Eye Protection Lamp': {
        'hand_start': 'Index finger IP joint flexion toward switch',
        'torso_start': 'Cervical rotation for visual confirmation',
        'leg_start': 'Lateral weight shift for reach optimization'
    },
    'Opening/Closing Curtains': {
        'hand_start': 'Digital flexion around curtain edge',
        'torso_start': 'Rotator cuff co-contraction for shoulder stability',
        'leg_start': 'Toe-off phase initiates body forward propulsion'
    },
    'Opening/Closing Windows': {
        'hand_start': 'Cylindrical grasp formation on handle',
        'torso_start': 'Latissimus dorsi pre-stretch for pulling',
        'leg_start': 'Weight transfer to contralateral limb'
    },
    'Typing': {
        'hand_start': 'Metacarpophalangeal joint positioning over keys',
        'torso_start': 'Scapular protraction for keyboard reach',
        'leg_start': 'Femoral internal rotation in seated position'
    },
    'Opening Envelope': {
        'hand_start': 'Thumb-forefinger web space manipulation',
        'torso_start': 'Sternal lift for improved visual axis',
        'leg_start': 'Quiet standing postural adjustments'
    },
    'Throwing Garbage': {
        'hand_start': 'Power grasp formation on trash item',
        'torso_start': 'Rotational torque generation in transverse plane',
        'leg_start': 'Stance phase initiation with lead leg'
    },
    'Picking Fruit': {
        'hand_start': 'Pre-shaping hand aperture to fruit size',
        'torso_start': 'Spinal lateral flexion toward target side',
        'leg_start': 'Single-leg balance activation'
    },
    'Picking Up Items': {
        'hand_start': 'Extrinsic finger flexor pre-activation',
        'torso_start': 'Hip hinge mechanics initiation',
        'leg_start': 'Hamstring eccentric loading'
    },
    'Answering Phone': {
        'hand_start': 'Radial deviation for device retrieval',
        'torso_start': 'Cervical lateral flexion for ear alignment',
        'leg_start': 'Automatic stepping pattern inhibition'
    },
    'Using Mouse': {
        'hand_start': 'Ulnar deviation for cursor positioning',
        'torso_start': 'Forward head posture onset',
        'leg_start': 'Ischial weight bearing in seated position'
    },
    'Wiping Table': {
        'hand_start': 'Palmar abrasion force application',
        'torso_start': 'Anterior deltoid activation for reach',
        'leg_start': 'Lateral weight shifting pattern'
    },
    'Writing on Blackboard': {
        'hand_start': 'Shoulder abduction to writing surface height',
        'torso_start': 'Scapular upward rotation for arm elevation',
        'leg_start': 'Bilateral heel elevation for reach'
    },
    'Washing Hands': {
        'hand_start': 'Proximal interphalangeal joint flexion under faucet',
        'torso_start': 'Lumbar stabilization against sink edge',
        'leg_start': 'Anticipatory postural adjustments'
    },
    'Using Phone': {
        'hand_start': 'Digital extensor activation for screen tap',
        'torso_start': 'Forward head posture initiation',
        'leg_start': 'Reduced base of support in standing'
    },
    'Reading': {
        'hand_start': 'Book opening with thenar eminence pressure',
        'torso_start': 'Thoracic extension for visual comfort',
        'leg_start': 'Lower extremity circulatory adjustments'
    },
    'Watering Plants': {
        'hand_start': 'Trigger pressurization initiation',
        'torso_start': 'Scapulohumeral rhythm coordination',
        'leg_start': 'Stride length adaptation to plant spacing'
    },
    'Walking to Bed': {
        'hand_start': 'Upper extremity reciprocal arm swing onset',
        'torso_start': 'Center of mass anterior translation',
        'leg_start': 'Initial contact phase (heel strike)'
    },
    'Walking to Chair': {
        'hand_start': 'Preparatory grasp formation for chair contact',
        'torso_start': 'Pelvic anteversion increases',
        'leg_start': 'Stance phase duration modulation'
    },
    'Walking to Cabinet': {
        'hand_start': 'Reach phase kinematic chain initiation',
        'torso_start': 'Trunk rotation toward target',
        'leg_start': 'Cadence adjustment for distance'
    },
    'Walking to Window': {
        'hand_start': 'Arm swing amplitude calibration',
        'torso_start': 'Visual focus alters gait parameters',
        'leg_start': 'Step width narrowing'
    },
    'Walking to Blackboard': {
        'hand_start': 'Chalk/marker transport grip formation',
        'torso_start': 'Anticipatory postural set',
        'leg_start': 'Acceleration phase ground reaction forces'
    },
    'Getting Out of Bed': {
        'hand_start': 'Push-off force generation through palms',
        'torso_start': 'Cervical erector spinae activation',
        'leg_start': 'Hip flexion angle reduction'
    },
    'Standing Up': {
        'hand_start': 'Grip force on armrests exceeds body weight 30%',
        'torso_start': 'Center of pressure shifts anteriorly',
        'leg_start': 'Knee extensor torque generation'
    },
    'Lying Down': {
        'hand_start': 'Upper extremity guidance phase',
        'torso_start': 'Eccentric control of spinal flexion',
        'leg_start': 'Hip extension angle gradual reduction'
    },
    'Standing Still': {
        'hand_start': 'Proprioceptive positioning calibration',
        'torso_start': 'Vestibular system input integration',
        'leg_start': 'Postural sway frequency <1Hz'
    },
    'Lying Still': {
        'hand_start': 'Gravitational force distribution on support surface',
        'torso_start': 'Respiratory diaphragm excursion onset',
        'leg_end': 'Muscle tone reduction below 10% MVC'
    },
    'Walking': {
        'hand_start': 'Contralateral arm swing initiation',
        'torso_start': 'Rotational momentum transfer preparation',
        'leg_start': 'Heel-strike impact (0.5-1.5 x body weight)'
    }
}


action_breakdown_end = {
    # 手部终止姿势 | 躯干稳定机制 | 腿部减速控制 
    'Stretching': {
        'hand_end': 'Proprioceptive feedback dampens muscle spindle activity',
        'torso_end': 'Paraspinal muscles return to resting tone',
        'leg_end': 'Ground reaction forces normalize to baseline'
    },
    'Pouring Water': {
        'hand_end': 'Grip force reduces below container slip threshold',
        'torso_end': 'Posterior chain muscles re-engage for upright posture',
        'leg_end': 'Bilateral weight distribution symmetry restored'
    },
    'Writing': {
        'hand_end': 'Extensor digitorum relaxes pen pressure',
        'torso_end': 'Thoracic erector spinae decrease activity',
        'leg_end': 'Lower extremity venous return resumes'
    },
    'Cutting Fruit': {
        'hand_end': 'Safety position achieved (blade edge control)',
        'torso_end': 'Intra-abdominal pressure releases',
        'leg_end': 'Hip adductor co-contraction ceases'
    },
    'Eating Fruit': {
        'hand_end': 'Mastication phase hand returns to neutral position',
        'torso_end': 'Hyolaryngeal complex descends post-swallow',
        'leg_end': 'Plantar pressure redistributes to midfoot'
    },
    'Taking Medicine': {
        'hand_end': 'Bottle closure tactile confirmation complete',
        'torso_end': 'Accessory breathing muscles relax',
        'leg_end': 'Static balance parameters return to baseline'
    },
    'Drinking Water': {
        'hand_end': 'Cup deceleration phase completes',
        'torso_end': 'Laryngeal elevation reverses',
        'leg_end': 'Quiet standing postural sway resumes'
    },
    'Sitting Down': {
        'hand_end': 'Upper extremity support force diminishes',
        'torso_end': 'Ischial tuberosity full weight acceptance',
        'leg_end': 'Knee flexion angle stabilizes at 90°'
    },
    'Turning On/Off Eye Protection Lamp': {
        'hand_end': 'Digital pressure releases from switch',
        'torso_end': 'Cervical spine returns to neutral alignment',
        'leg_end': 'Base of support narrows to standing width'
    },
    'Opening/Closing Curtains': {
        'hand_end': 'Curtain material tension equalizes',
        'torso_end': 'Rotator cuff force couple rebalances',
        'leg_end': 'Stance phase lower limb recovers'
    },
    'Opening/Closing Windows': {
        'hand_end': 'Handle returns to neutral detent position',
        'torso_end': 'Latissimus dorsi reciprocal inhibition',
        'leg_end': 'Bilateral weight bearing symmetry restored'
    },
    'Typing': {
        'hand_end': 'Key rebound damping completes',
        'torso_end': 'Scapular retractor activation begins',
        'leg_end': 'Seated pressure redistribution occurs'
    },
    'Opening Envelope': {
        'hand_end': 'Flap separation verified tactiley',
        'torso_end': 'Vertebral column re-establishes neutral',
        'leg_end': 'Standing postural adjustments complete'
    },
    'Throwing Garbage': {
        'hand_end': 'Follow-through phase dissipates kinetic energy',
        'torso_end': 'Rotational deceleration completes',
        'leg_end': 'Step recovery phase achieves balance'
    },
    'Picking Fruit': {
        'hand_end': 'Fruit-stem separation confirmed',
        'torso_end': 'Lateral flexion reverses',
        'leg_end': 'Single-leg support transitions to bilateral'
    },
    'Picking Up Items': {
        'hand_end': 'Object secure in palmar containment',
        'torso_end': 'Hip extension returns to neutral',
        'leg_end': 'Knee extension completes lift phase'
    },
    'Answering Phone': {
        'hand_end': 'Device reaches auditory target zone',
        'torso_end': 'Cervical alignment optimizes for sound conduction',
        'leg_end': 'Automatic stepping patterns re-initiate'
    },
    'Using Mouse': {
        'hand_end': 'Cursor reaches target coordinates',
        'torso_end': 'Forward head moment arm reduces',
        'leg_end': 'Lower limb venous stasis resumes'
    },
    'Wiping Table': {
        'hand_end': 'Cleaning surface verified visually',
        'torso_end': 'Anterior muscle chain relaxes',
        'leg_end': 'Lateral weight shifts cease'
    },
    'Writing on Blackboard': {
        'hand_end': 'Chalk dust release completes',
        'torso_end': 'Scapular downward rotation occurs',
        'leg_end': 'Heel contact re-establishes'
    },
    'Washing Hands': {
        'hand_end': 'Soap residue fully rinsed',
        'torso_end': 'Lumbar loading decreases',
        'leg_end': 'Antigravity muscle activity normalizes'
    },
    'Using Phone': {
        'hand_end': 'Touch input feedback received',
        'torso_end': 'Cervical extensor muscles re-engage',
        'leg_end': 'Base of support widens for stability'
    },
    'Reading': {
        'hand_end': 'Page corner secure in book spine',
        'torso_end': 'Thoracic flexion reduces to neutral',
        'leg_end': 'Lower extremity circulation optimizes'
    },
    'Watering Plants': {
        'hand_end': 'Trigger pressure releases completely',
        'torso_end': 'Shoulder complex returns to resting',
        'leg_end': 'Stride frequency decreases to standing'
    },
    'Walking to Bed': {
        'hand_end': 'Bed surface contact confirmed',
        'torso_end': 'Center of mass stabilizes over base',
        'leg_end': 'Terminal stance phase completes'
    },
    'Walking to Chair': {
        'hand_end': 'Chair armrest contact established',
        'torso_end': 'Pelvic tilt neutralizes',
        'leg_end': 'Step-to pattern concludes'
    },
    'Walking to Cabinet': {
        'hand_end': 'Cabinet handle within grasp envelope',
        'torso_end': 'Trunk rotation halts at target',
        'leg_end': 'Deceleration phase completes'
    },
    'Walking to Window': {
        'hand_end': 'Window frame within tactile range',
        'torso_end': 'Visual focus locks on target',
        'leg_end': 'Terminal double support phase'
    },
    'Walking to Blackboard': {
        'hand_end': 'Writing implement positioned correctly',
        'torso_end': 'Forward lean reduces to standing',
        'leg_end': 'Propulsive phase forces dissipate'
    },
    'Getting Out of Bed': {
        'hand_end': 'Full upright posture achieved',
        'torso_end': 'Vertical center of mass stabilized',
        'leg_end': 'Feet flat floor contact confirmed'
    },
    'Standing Up': {
        'hand_end': 'Armrest grip force drops below 10N',
        'torso_end': 'Vertical ground reaction force equals body weight',
        'leg_end': 'Knee hyperextension prevented'
    },
    'Lying Down': {
        'hand_end': 'Support surface full contact achieved',
        'torso_end': 'Spinal curves conform to surface',
        'leg_end': 'Lower extremity muscle activity <5% MVC'
    },
    'Standing Still': {
        'hand_end': 'Proprioceptive drift corrected',
        'torso_end': 'Vestibulo-ocular reflex stabilizes gaze',
        'leg_end': 'Center of pressure within stability limits'
    },
    'Lying Still': {
        'hand_end': 'Palmar contact pressure equalizes',
        'torso_end': 'Respiratory rate enters resting rhythm',
        'leg_end': 'Gravity line aligns with support surface'
    },
    'Walking': {
        'hand_end': 'Arm swing amplitude dampens',
        'torso_end': 'Rotational momentum fully dissipated',
        'leg_end': 'Terminal double support phase concludes'
    }
}

id_to_attribute_start = {k: action_breakdown_start[v] for k, v in new_id_to_action.items()}
id_to_attribute_end = {k: action_breakdown_end[v] for k, v in new_id_to_action.items()}


#---------------根据描述文本找到对应的标签-------------#
# doing_action = {k for k, v in action_breakdown_start.items()}
# done_action = {k for k, v in action_breakdown_end.items()}
# todo_action = {k for k, v in action_breakdown_start.items()}

values = list(id_to_attribute.values()) + list(id_to_attribute_start.values()) + list(id_to_attribute_end.values())
attribute_to_newIdx = {index: value for index, value in enumerate(values)}
# newIdx_to_attribute = {v : k for k, v in attribute_to_newIdx}


# attribute_start_to_action = {v : k for k, v in action_breakdown_start}
# attribute_end_to_action = {v : k for k, v in action_breakdown_end}
# attribute_to_action = {v : k for k, v in action_attribute}