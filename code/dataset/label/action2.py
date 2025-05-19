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
    'Stretching': 'A person is stretching with arms raised upwards and fingers extended, back slightly arched, and legs apart for balance',
    'Pouring Water': 'A person is pouring water with a cup/pot gripped and wrist tilted, torso leaning forward slightly, and legs in stationary stance',
    'Writing': 'A person is writing with a pen held in precise finger movements, torso bent forward at desk, and legs seated/standing still',
    'Cutting Fruit': 'A person is cutting fruit with a knife in dominant hand and fruit stabilized with other hand, torso leaning over cutting board, and legs standing in place',
    'Eating Fruit': 'A person is eating fruit with food brought to mouth in chewing motion, torso in upright sitting position, and legs relaxed under table',
    'Taking Medicine': 'A person is taking medicine with pills picked up and medicine bottle opened, torso in slight forward lean, and legs standing near cabinet',
    'Drinking Water': 'A person is drinking water with cup lifted to lips for swallowing, head tilted back, and legs stationary in standing/sitting position',
    'Sitting Down': 'A person is sitting down with hands pushing on armrests/desk, torso lowering body, and legs bending knees with position adjustment',
    'Turning On/Off Eye Protection Lamp': 'A person is turning on/off eye protection lamp with switch pressed, torso facing lamp, and legs standing within reach',
    'Opening/Closing Curtains': 'A person is opening/closing curtains with curtain fabric/cord grasped, arms extended sideways, and legs potentially using step stool',
    'Opening/Closing Windows': 'A person is opening/closing windows with handle turned and window pushed/pulled, torso leaning toward window, and legs shifting weight for leverage',
    'Typing': 'A person is typing with fingers pressing keys, torso in upright posture, and legs seated in position',
    'Opening Envelope': 'A person is opening envelope with finger sliding along seal, envelope held at chest height, and legs standing stationary',
    'Throwing Garbage': 'A person is throwing garbage with trash grasped and arm swung toward bin, torso twisting at waist, and legs stepping toward bin',
    'Picking Fruit': 'A person is picking fruit with plucking motion using hand-eye coordination, torso reaching upward, and legs tip-toeing when needed',
    'Picking Up Items': 'A person is picking up items with fingers bent to grasp, torso squatting/bending at waist, and legs flexing knees for lower items',
    'Answering Phone': 'A person is answering phone with device held to ear, torso in upright posture, and legs potentially pacing',
    'Using Mouse': 'A person is using mouse with wrist moving and buttons clicked, shoulders positioned forward, and legs stationary in seated position',
    'Wiping Table': 'A person is wiping table with circular scrubbing motion, torso leaning over surface, and legs side-stepping along table',
    'Writing on Blackboard': 'A person is writing on blackboard with chalk gripped and arm elevated, torso facing board, and legs standing with weight shifts',
    'Washing Hands': 'A person is washing hands with palms rubbed and fingers interlaced, torso bent over sink, and legs standing at basin',
    'Using Phone': 'A person is using phone with screen tapped/swiped, neck flexed forward, and legs stationary or pacing',
    'Reading': 'A person is reading with pages turned and book held, torso in relaxed sitting posture, and legs possibly crossed',
    'Watering Plants': 'A person is watering plants with trigger squeezed and spout aimed, arms elevated, and legs walking between plants',
    'Walking to Bed': 'A person is walking to bed with covers possibly pulled back, torso with forward momentum, and legs in alternating strides',
    'Walking to Chair': 'A person is walking to chair with arms swinging for balance, torso upright, and legs taking controlled steps',
    'Walking to Cabinet': 'A person is walking to cabinet with door potentially opened, torso oriented toward target, and legs moving directionally',
    'Walking to Window': 'A person is walking to window with arm swings coordinated, torso facing forward, and legs maintaining steady gait',
    'Walking to Blackboard': 'A person is walking to blackboard with chalk/marker carried, torso leaning forward during motion, and legs accelerating steps',
    'Getting Out of Bed': 'A person is getting out of bed with hands pushing on mattress, torso engaging core to sit up, and legs swinging over edge',
    'Standing Up': 'A person is standing up with hands pushing on knees/armrests, torso vertically elevated, and legs extending knees',
    'Lying Down': 'A person is lying down with body lowering guided by hands, torso in controlled recline, and legs lifted sequentially',
    'Standing Still': 'A person is standing still with hands relaxed at sides, torso in neutral spinal alignment, and legs distributing weight evenly',
    'Lying Still': 'A person is lying still with hands resting on surface, torso in full contact with bed, and legs extended straight',
    'Walking': 'A person is walking with arms swinging naturally, torso slightly leaning forward, and legs propelling alternately'
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
    'Stretching': 'A person is starting to stretch with arms initiating upward trajectory from rest, intercostal muscles engaging for ribcage expansion, and heel contact area increasing for base support',
    'Pouring Water': 'A person is starting to pour water with thenar eminence contacting container handle, sternocleidomastoid stabilizing head, and subtalar joint pronating for weight shift',
    'Writing': 'A person is starting to write with tripod grasp forming on pen, thoracic kyphosis increasing for desk proximity, and ischial pressure distributing in seated position',
    'Cutting Fruit': 'A person is starting to cut fruit with palm-force closure on knife handle, abdominal muscles bracing core, and hips externally rotating for wide stance',
    'Eating Fruit': 'A person is starting to eat fruit with grip adjusting prehensilely, cervical flexing to lower head, and plantar pressure shifting to metatarsal heads',
    'Taking Medicine': 'A person is starting to take medicine with pincer grasp on pill, sternum elevating for breathing, and tibialis anterior activating for balance',
    'Drinking Water': 'A person is starting to drink water with thenar-hypothenar compressing cup, hyoid elevating for swallowing, and gastrocnemius activating in quiet stance',
    'Sitting Down': 'A person is starting to sit down with palms contacting support surface, pelvis posteriorly tilting, and quadriceps eccentrically loading',
    'Turning On/Off Eye Protection Lamp': 'A person is starting to toggle lamp with index finger flexing toward switch, neck rotating for visibility, and weight shifting laterally for reach',
    'Opening/Closing Curtains': 'A person is starting to adjust curtains with fingers flexing on fabric, rotator cuff co-contracting, and toes pushing off for forward motion',
    'Opening/Closing Windows': 'A person is starting to operate window with cylindrical grasp on handle, latissimus dorsi prestretching, and weight transferring to opposite leg',
    'Typing': 'A person is starting to type with MCP joints positioning over keys, scapulae protracting, and femurs internally rotating when seated',
    'Opening Envelope': 'A person is starting to open envelope with thumb-web manipulation, sternum lifting for visibility, and posture adjusting in quiet standing',
    'Throwing Garbage': 'A person is starting to throw trash with power grasp formation, torso generating rotational torque, and lead leg initiating stance phase',
    'Picking Fruit': 'A person is starting to pick fruit with hand preshaping to size, spine laterally flexing, and single-leg balance activating',
    'Picking Up Items': 'A person is starting to pick objects with finger flexors pre-activating, hip hinging initiating, and hamstrings loading eccentrically',
    'Answering Phone': 'A person is starting to answer phone with radial deviating grasp, neck laterally flexing for ear alignment, and stepping patterns inhibiting',
    'Using Mouse': 'A person is starting to use mouse with ulnar deviating position, forward head posture beginning, and ischial weight bearing increasing',
    'Wiping Table': 'A person is starting to wipe table with palmar abrasion force, anterior deltoids reaching, and weight shifting laterally',
    'Writing on Blackboard': 'A person is starting to write on blackboard with shoulder abducting, scapulae upwardly rotating, and heels elevating for reach',
    'Washing Hands': 'A person is starting to wash hands with PIP joints flexing under water, lumbar stabilizing against sink, and posture adjusting anticipatorily',
    'Using Phone': 'A person is starting to use phone with digital extensors activating, forward head posture initiating, and standing base narrowing',
    'Reading': 'A person is starting to read with thenar pressure opening book, thoracic extending, and leg circulation adjusting',
    'Watering Plants': 'A person is starting to water plants with trigger pressurizing, scapulohumeral rhythm coordinating, and stride adapting to plant spacing',
    'Walking to Bed': 'A person is starting to walk to bed with arms swinging reciprocally, center of mass shifting forward, and heel striking initially',
    'Walking to Chair': 'A person is starting to walk to chair with grasp pre-forming, pelvis anteverting, and stance phase modulating',
    'Walking to Cabinet': 'A person is starting to walk to cabinet with reach kinematics initiating, trunk rotating, and cadence adjusting for distance',
    'Walking to Window': 'A person is starting to walk to window with arm swing calibrating, gaze altering gait, and step width narrowing',
    'Walking to Blackboard': 'A person is starting to walk to blackboard with chalk grip forming, posture anticipating, and ground forces increasing',
    'Getting Out of Bed': 'A person is starting to get up with palms pushing off mattress, cervical extensors activating, and hip flexion decreasing',
    'Standing Up': 'A person is starting to stand up with grip exceeding 30% body weight, center of pressure shifting, and knee torque generating',
    'Lying Down': 'A person is starting to lie down with arms guiding descent, spine controlling flexion, and hip extension reducing',
    'Standing Still': 'A person is starting to stand still with proprioception calibrating, vestibular inputs integrating, and sway maintaining <1Hz',
    'Lying Still': 'A person is starting to lie still with gravity distributing weight, diaphragm excursing, and muscle tone reducing below 10% MVC',
    'Walking': 'A person is starting to walk with contralateral arm swinging, rotational momentum preparing, and heel impacting at 0.5-1.5BW'
}

action_breakdown_end = {
    'Stretching': 'A person is finishing stretching with proprioception reducing spindle activity, paraspinals returning to rest, and ground forces normalizing',
    'Pouring Water': 'A person is finishing pouring water with grip below slip threshold, posterior chain re-engaging, and weight distribution symmetrizing',
    'Writing': 'A person is finishing writing with pen pressure relaxing, thoracic erectors decreasing activity, and venous return resuming in legs',
    'Cutting Fruit': 'A person is finishing cutting fruit with blade secured, intra-abdominal pressure releasing, and hip adductors ceasing co-contraction',
    'Eating Fruit': 'A person is finishing eating fruit with hands neutralized, hyolaryngeal complex descending, and plantar pressure shifting midfoot',
    'Taking Medicine': 'A person is finishing taking medicine with bottle closed, accessory muscles relaxing, and balance parameters normalizing',
    'Drinking Water': 'A person is finishing drinking water with cup deceleration complete, larynx descending, and postural sway returning',
    'Sitting Down': 'A person is finishing sitting down with arm support diminishing, ischial weight fully accepted, and knees stabilized at 90°',
    'Turning On/Off Eye Protection Lamp': 'A person is finishing toggling lamp with pressure released, neck realigned, and base narrowed to standing',
    'Opening/Closing Curtains': 'A person is finishing curtain adjustment with tension equalized, rotator cuff rebalanced, and stance limb recovered',
    'Opening/Closing Windows': 'A person is finishing window operation with handle neutralized, lats inhibited, and weight symmetry restored',
    'Typing': 'A person is finishing typing with key rebound damped, scapulae retracting, and seated pressure redistributing',
    'Opening Envelope': 'A person is finishing opening envelope with flap separated, spine neutralized, and standing adjustments completed',
    'Throwing Garbage': 'A person is finishing throwing trash with energy dissipated, rotation stopped, and step recovery balanced',
    'Picking Fruit': 'A person is finishing picking fruit with stem detached, lateral flexion reversed, and support transitioned to bilateral',
    'Picking Up Items': 'A person is finishing picking objects with item secured, hips neutralized, and knee extension completed',
    'Answering Phone': 'A person is finishing answering phone with device positioned, neck aligned, and stepping patterns resumed',
    'Using Mouse': 'A person is finishing using mouse with cursor targeted, head moment reduced, and venous stasis returning',
    'Wiping Table': 'A person is finishing wiping table with surface cleaned, anterior muscles relaxed, and lateral shifts ceased',
    'Writing on Blackboard': 'A person is finishing blackboard writing with chalk dust cleared, scapulae down-rotated, and heel contact re-established',
    'Washing Hands': 'A person is finishing washing hands with soap rinsed, lumbar load decreased, and antigravity activity normalized',
    'Using Phone': 'A person is finishing phone use with touch feedback received, cervical extensors engaged, and base widened',
    'Reading': 'A person is finishing reading with page secured, thoracic flexion neutralized, and leg circulation optimized',
    'Watering Plants': 'A person is finishing watering plants with trigger released, shoulders resting, and stride frequency decreased',
    'Walking to Bed': 'A person is finishing walking to bed with contact confirmed, center stabilized, and terminal stance completed',
    'Walking to Chair': 'A person is finishing walking to chair with armrest contact made, pelvis neutral, and step-to pattern concluded',
    'Walking to Cabinet': 'A person is finishing walking to cabinet with handle in range, trunk rotation halted, and deceleration done',
    'Walking to Window': 'A person is finishing walking to window with frame in reach, gaze locked, and double support phase entered',
    'Walking to Blackboard': 'A person is finishing walking to blackboard with chalk positioned, lean reduced, and propulsion dissipated',
    'Getting Out of Bed': 'A person is finishing getting up with posture upright, center stabilized, and feet flat on floor',
    'Standing Up': 'A person is finishing standing up with grip <10N, vertical force equalized, and hyperextension prevented',
    'Lying Down': 'A person is finishing lying down with full contact achieved, spine conformed, and leg activity <5% MVC',
    'Standing Still': 'A person is finishing standing still with drift corrected, gaze stabilized, and center within limits',
    'Lying Still': 'A person is finishing lying still with pressure equalized, breathing restful, and gravity line aligned',
    'Walking': 'A person is finishing walking with swing dampened, momentum dissipated, and final support phase concluded'
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