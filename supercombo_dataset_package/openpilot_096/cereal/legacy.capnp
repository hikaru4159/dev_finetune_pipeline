# Dummy struct for Legacy.OrbObservation to satisfy log.capnp
struct OrbObservation {
  dummy @0 :Int64;
}

# Dummy struct for Legacy.OrbOdometry to satisfy log.capnp
struct OrbOdometry {
  dummy @0 :Int64;
}

# Dummy struct for Legacy.TrafficEvent to satisfy log.capnp
struct TrafficEvent {
  dummy @0 :Int64;
}
# Dummy struct for Legacy.NavStatus to satisfy log.capnp
struct NavStatus {
  dummy @0 :Int64;
}
# Dummy struct for Legacy.LidarPts to satisfy log.capnp
struct LidarPts {
  dummy @0 :Int64;
}
# Dummy struct for Legacy.AndroidGnss to satisfy log.capnp
struct AndroidGnss {
  dummy @0 :Int64;
}
using Cxx = import "./include/c++.capnp";
$Cxx.namespace("cereal");
@0x80ef1ec4889c2a63;
# legacy.capnp: a home for deprecated structs
struct LogRotate @0x9811e1f38f62f2d1 {
  segmentNum @0 :Int32;
  path @1 :Text;
}

struct LiveUI @0xc08240f996aefced {
  rearViewCam @0 :Bool;
  alertText1 @1 :Text;
  alertText2 @2 :Text;
  awarenessStatus @3 :Float32;
}

struct UiLayoutState @0x88dcce08ad29dda0 {
  activeApp @0 :App;
  sidebarCollapsed @1 :Bool;
  mapEnabled @2 :Bool;
  mockEngaged @3 :Bool;

  enum App @0x9917470acf94d285 {
    home @0;
    music @1;
    nav @2;
    settings @3;
    none @4;
  }
}

struct OrbslamCorrection @0x8afd33dc9b35e1aa {
  correctionMonoTime @0 :UInt64;
  prePositionECEF @1 :List(Float64);
  postPositionECEF @2 :List(Float64);
  prePoseQuatECEF @3 :List(Float32);
  postPoseQuatECEF @4 :List(Float32);
  numInliers @5 :UInt32;
}

struct EthernetPacket @0xa99a9d5b33cf5859 {
  pkt @0 :Data;
  ts @1 :Float32;
}

struct CellInfo @0xcff7566681c277ce {
  timestamp @0 :UInt64;
  repr @1 :Text; # android toString() for now
}

struct WifiScan @0xd4df5a192382ba0b {
  bssid @0 :Text;
  ssid @1 :Text;
  capabilities @2 :Text;
  frequency @3 :Int32;
  level @4 :Int32;
  timestamp @5 :Int64;

  centerFreq0 @6 :Int32;
  centerFreq1 @7 :Int32;
  channelWidth @8 :ChannelWidth;
  operatorFriendlyName @9 :Text;
  venueName @10 :Text;
  is80211mcResponder @11 :Bool;
  passpoint @12 :Bool;

  distanceCm @13 :Int32;
  distanceSdCm @14 :Int32;

  enum ChannelWidth @0xcb6a279f015f6b51 {
    w20Mhz @0;
    w40Mhz @1;
    w80Mhz @2;
    w160Mhz @3;
    w80Plus80Mhz @4;
  }
}

struct LiveEventData @0x94b7baa90c5c321e {
  name @0 :Text;
  value @1 :Int32;
}

struct ModelData @0xb8aad62cffef28a9 {
  frameId @0 :UInt32;
  frameAge @12 :UInt32;
  frameDropPerc @13 :Float32;
  timestampEof @9 :UInt64;
  modelExecutionTime @14 :Float32;
  gpuExecutionTime @16 :Float32;
  rawPred @15 :Data;

  path @1 :PathData;
  leftLane @2 :PathData;
  rightLane @3 :PathData;
  lead @4 :LeadData;
  freePath @6 :List(Float32);

  settings @5 :ModelSettings;
  leadFuture @7 :LeadData;
  speed @8 :List(Float32);
  meta @10 :MetaData;
  longitudinal @11 :LongitudinalData;

  struct PathData @0x8817eeea389e9f08 {
    points @0 :List(Float32);
    prob @1 :Float32;
    std @2 :Float32;
    stds @3 :List(Float32);
    poly @4 :List(Float32);
    validLen @5 :Float32;
  }

  struct LeadData @0xd1c9bef96d26fa91 {
    dist @0 :Float32;
    prob @1 :Float32;
    std @2 :Float32;
    relVel @3 :Float32;
    relVelStd @4 :Float32;
    relY @5 :Float32;
    relYStd @6 :Float32;
    relA @7 :Float32;
    relAStd @8 :Float32;
  }

  struct ModelSettings @0xa26e3710efd3e914 {
    bigBoxX @0 :UInt16;
    bigBoxY @1 :UInt16;
    bigBoxWidth @2 :UInt16;
    bigBoxHeight @3 :UInt16;
    boxProjection @4 :List(Float32);
    yuvCorrection @5 :List(Float32);
    inputTransform @6 :List(Float32);
  }

  struct MetaData @0x9744f25fb60f2bf8 {
    engagedProb @0 :Float32;
    desirePrediction @1 :List(Float32);
    brakeDisengageProb @2 :Float32;
    gasDisengageProb @3 :Float32;
    steerOverrideProb @4 :Float32;
    desireState @5 :List(Float32);
  }

  struct LongitudinalData @0xf98f999c6a071122 {
    distances @2 :List(Float32);
    speeds @0 :List(Float32);
    accelerations @1 :List(Float32);
  }
}

struct ECEFPoint @0xc25bbbd524983447 {
  x @0 :Float64;
  y @1 :Float64;
  z @2 :Float64;
}

struct ECEFPointDEPRECATED @0xe10e21168db0c7f7 {
  x @0 :Float32;
  y @1 :Float32;
  z @2 :Float32;
}

struct GPSPlannerPoints @0xab54c59699f8f9f3 {
  curPosDEPRECATED @0 :ECEFPointDEPRECATED;
  pointsDEPRECATED @1 :List(ECEFPointDEPRECATED);
  curPos @6 :ECEFPoint;
  points @7 :List(ECEFPoint);
  valid @2 :Bool;
  trackName @3 :Text;
  speedLimit @4 :Float32;
  accelTarget @5 :Float32;
}

struct GPSPlannerPlan @0xf5ad1d90cdc1dd6b {
  valid @0 :Bool;
  poly @1 :List(Float32);
  trackName @2 :Text;
  speed @3 :Float32;
  acceleration @4 :Float32;
  pointsDEPRECATED @5 :List(ECEFPointDEPRECATED);
  points @6 :List(ECEFPoint);
  xLookahead @7 :Float32;
}

struct UiNavigationEvent @0x90c8426c3eaddd3b {
  type @0: Type;
  status @1: Status;
  distanceTo @2: Float32;
  endRoadPointDEPRECATED @3: ECEFPointDEPRECATED;
  endRoadPoint @4: ECEFPoint;

  enum Type @0xe8db07dcf8fcea05 {
    none @0;
    laneChangeLeft @1;
    laneChangeRight @2;
    mergeLeft @3;
    mergeRight @4;
    turnLeft @5;
    turnRight @6;
  }

  enum Status @0xb9aa88c75ef99a1f {
    none @0;
    passive @1;
    approaching @2;
    active @3;
  }
}

struct LiveLocationData @0xb99b2bc7a57e8128 {
  status @0 :UInt8;

  # 3D fix
  lat @1 :Float64;
  lon @2 :Float64;
  alt @3 :Float32;     # m

  # speed
  speed @4 :Float32;   # m/s

  # NED velocity components
  vNED @5 :List(Float32);

  # roll, pitch, heading (x,y,z)
  roll @6 :Float32;     # WRT to center of earth?
  pitch @7 :Float32;    # WRT to center of earth?
  heading @8 :Float32;  # WRT to north?

  # what are these?
  wanderAngle @9 :Float32;
  trackAngle @10 :Float32;

  # car frame -- https://upload.wikimedia.org/wikipedia/commons/f/f5/RPY_angles_of_cars.png

  # gyro, in car frame, deg/s
  gyro @11 :List(Float32);

  # accel, in car frame, m/s^2
  accel @12 :List(Float32);

  accuracy @13 :Accuracy;

  source @14 :SensorSource;
  # if we are fixing a location in the past
  fixMonoTime @15 :UInt64;

  gpsWeek @16 :Int32;
  timeOfWeek @17 :Float64;

  positionECEF @18 :List(Float64);
  poseQuatECEF @19 :List(Float32);
  pitchCalibration @20 :Float32;
  yawCalibration @21 :Float32;
  imuFrame @22 :List(Float32);

  struct Accuracy @0x943dc4625473b03f {
    pNEDError @0 :List(Float32);
    vNEDError @1 :List(Float32);
    rollError @2 :Float32;
    pitchError @3 :Float32;
    headingError @4 :Float32;
    ellipsoidSemiMajorError @5 :Float32;
    ellipsoidSemiMinorError @6 :Float32;
    ellipsoidOrientationError @7 :Float32;
  }

  enum SensorSource @0xc871d3cc252af657 {
    applanix @0;
    kalman @1;
    orbslam @2;
    timing @3;
    dummy @4;
  }
}

# Minimal placeholder to satisfy references from log.capnp
struct CalibrationFeatures {
  # added as an empty placeholder; real fields are not required for current extraction
}

# Dummy struct for Legacy.NavUpdate to satisfy log.capnp
struct NavUpdate {
  # dummy field
  dummy @0 :Int64;
}

# Dummy struct for Legacy.OrbFeatures to satisfy log.capnp
struct OrbFeatures {
  dummy @0 :Int64;
}
# Dummy struct for Legacy.OrbKeyFrame to satisfy log.capnp
struct OrbKeyFrame {
  dummy @0 :Int64;
}
# Dummy struct for Legacy.OrbFeaturesSummary to satisfy log.capnp
struct OrbFeaturesSummary {
  dummy @0 :Int64;
}
# Dummy struct for Legacy.KalmanOdometry to satisfy log.capnp
struct KalmanOdometry {
  dummy @0 :Int64;
}




