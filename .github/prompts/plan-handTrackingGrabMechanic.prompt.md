# Plan: Hand Tracking + Grab Mechanic

## TL;DR
Add `XR_EXT_hand_tracking` to `XrSystem`, expose 26 joint positions per hand,
implement fist-gesture detection, add a dedicated per-hand `grabAction` for
controller fallback, and wire a grab mechanic in `main.cpp` so the player can
grab the single scene object by closing their fist (hand tracking) or pressing
the trigger (controllers) near the object — both hands supported.

---

## Phase 1 — `XrSystem`: Hand Tracking Extension Support

**1.1 Optional extension request** (`src/xr.cpp`: `create()`)
- Call `xrEnumerateInstanceExtensionProperties` to check if `XR_EXT_HAND_TRACKING_EXTENSION_NAME` is available before requesting it.
- Store `bool handTrackingExtEnabled`.

**1.2 Load function pointers** (`src/include/xr.hpp`, `src/xr.cpp`)
- Add three private members: `PFN_xrCreateHandTrackerEXT pfnCreateHandTracker{}`, `PFN_xrDestroyHandTrackerEXT pfnDestroyHandTracker{}`, `PFN_xrLocateHandJointsEXT pfnLocateHandJoints{}`.
- Load them via `xrGetInstanceProcAddr` immediately after `xrCreateInstance`.

**1.3 System capability check** (`src/xr.cpp`: `createSession()`)
- Chain `XrSystemHandTrackingPropertiesEXT` onto `XrSystemProperties` before calling `xrGetSystemProperties`.
- Store `bool handTrackingSupported`.

**1.4 Create/destroy hand trackers** (`src/xr.cpp`)
- Private `std::array<XrHandTrackerEXT, 2> handTrackers{XR_NULL_HANDLE, XR_NULL_HANDLE}`.
- Private `createHandTrackers()`: call `xrCreateHandTrackerEXT` for Left and Right; called at end of `createSession()` when both `handTrackingExtEnabled` and `handTrackingSupported` are true.
- Destroy trackers in `shutdownGraphics()` before destroying the session.

**1.5 Per-frame joint location** (`src/xr.cpp`: `beginFrame()`)
- New public struct `HandJointData` in `src/include/xr.hpp`:
  `bool isTracked = false;` + `std::array<XrHandJointLocationEXT, XR_HAND_JOINT_COUNT_EXT> joints{};`
- Private `std::array<HandJointData, 2> jointData{}`.
- Private `locateHandJoints(XrTime displayTime)` called inside `beginFrame()` after `xrLocateViews`. Uses `XrHandJointsLocateInfoEXT{baseSpace=appSpace, time=displayTime}` and `XrHandJointLocationsEXT{isActive, jointCount=26}`. No-op when trackers are null.
- Public accessor: `const HandJointData& getHandJoints(Hand h) const`.

**1.6 Per-hand grab action** (`src/xr.cpp`: `createInput()`)
- New `XrAction grabAction = XR_NULL_HANDLE` (BOOLEAN_INPUT, **per-hand subactions** — `true`).
- Add left- and right-hand trigger bindings to all four controller profiles:
  - Valve Index: `/user/hand/{left,right}/input/trigger/click`
  - Oculus Touch: `/user/hand/{left,right}/input/trigger/value`
  - HTC Vive: `/user/hand/{left,right}/input/trigger/click`
  - KHR Simple: `/user/hand/{left,right}/input/select/click`
- Destroy in `destroyInput()`.

**1.7 Grab gesture helper** (`src/xr.cpp`)
- New public `bool isGrabGesture(Hand h) const`:
  - If `getHandJoints(h).isTracked`: return `true` when `INDEX_TIP` (joint 10) and `MIDDLE_TIP` (joint 15) are both within **0.065 m** of `PALM` (joint 0).
  - Controller fallback (both hands): query `grabAction` with `subactionPath = handPaths[i]`; return its boolean state. Works for both hands even when hand tracking is entirely absent.
- Also expose `bool handTrackingAvailable() const { return handTrackingSupported; }` for ImGui diagnostics.

---

## Phase 2 — Grab Mechanic in `main.cpp`

**2.1 Helper** — add `placeModelAt(pos, scale, center)` to anonymous namespace;
existing `placeModel` delegates to it.

**2.2 Grab state** — add locals:
- `glm::vec3 objectWorldPos` = initial `worldPos` (`glm::vec3(0,1.2,-1.5)`)
- `float objectScale = 0.5f / model.radius`
- `bool objectGrabbed = false`, `XrSystem::Hand grabbingHand`, `glm::vec3 grabHandOffset`

**2.3 Per-frame logic** (inside `if (xrRunning)` after `syncActions()`):
- *Grab start* (objectGrabbed == false): for each hand —
  - Get hand position: prefer `PALM` joint from `getHandJoints()`; fallback to `xr.handPose()` translation (controller grip).
  - If `distance(handPos, objectWorldPos) < 0.35f` AND `xr.isGrabGesture(h)`: record offset, set `objectGrabbed = true`. Break.
- *Hold*: `objectWorldPos = handPos + grabHandOffset`; call `renderer.setObjectTransform(placeModelAt(...))`.
- *Release*: when `!isGrabGesture(grabbingHand)`, set `objectGrabbed = false`.

**2.4 ImGui** — add "Grab" section showing `handTrackingAvailable()` and grab state.

---

## Relevant Files

| File | Changes |
|------|---------|
| `src/include/xr.hpp` | `HandJointData` struct, fn-pointer members, tracker handles, `grabAction`, new methods + flags |
| `src/xr.cpp` | All Phase 1 implementation |
| `src/main.cpp` | `placeModelAt` helper, grab state + logic, ImGui grab section |

---

## Verification

1. `cmake --build build` — no errors (fn pointers loaded at runtime).
2. **VR + hand tracking**: close fist within 35 cm → object follows; open → drops.
3. **VR, no hand tracking**: trigger button grabs with **either** hand.
4. **Desktop**: app starts clean, no crash.

---

## Decisions / Scope

- **Hand models not included** (requires multi-object rendering + mesh assets) — user notified.
- **Position-only grab**, no rotation transfer — fits Force telekinesis.
- **Single-hand grab** — first hand triggering gesture wins.
- Grab range **0.35 m**, fist threshold **0.065 m** — named constants.
- Existing `selectAction` (right-hand only) is kept unchanged; `grabAction` is the new per-hand action.
