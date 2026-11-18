// ─── Types ───────────────────────────────────────────────────────────────────

export type Site = "basspro.com" | "cabelas.com" | "basspro.ca" | "mackspw.com";

export type PromotionStatus =
  | "draft"
  | "in_review"
  | "approved"
  | "active"
  | "suspended"
  | "ended"
  | "archived";

export type RewardKind = "PERCENT" | "AMOUNT_OFF" | "FIXED_PRICE" | "FREE_ITEM";
export type TargetLevel = "ITEM" | "ORDER" | "SHIPPING";

export type Role =
  | "merchandiser"
  | "approver"
  | "site_admin"
  | "super_admin"
  | "viewer_qa"
  | "auditor";

export interface Template {
  code: string;
  name: string;
  reward: RewardKind;
  target: string;
  level: TargetLevel;
}

export interface Promotion {
  id: string;
  name: string;
  templateCode: string;
  templateName: string;
  status: PromotionStatus;
  sites: Site[];
  campaignId?: string;
  campaignName?: string;
  priority: number;
  version: number;
  startsAt: string;
  endsAt: string;
  createdBy: string;
  createdAt: string;
  updatedAt: string;
  reward: {
    kind: RewardKind;
    value: number;
    maxDiscount?: number;
  };
  conditions: {
    categories?: string[];
    skus?: string[];
    minQty?: number;
    minSpend?: number;
    segments?: string[];
    channels?: string[];
  };
  combinationMode: "STACKABLE" | "EXCLUSIVE" | "BEST_DEAL" | "ONE_PER_LEVEL";
  codes?: string[];
  limits?: {
    totalRedemptions?: number;
    perCustomer?: number;
    usedCount?: number;
  };
}

export interface Campaign {
  id: string;
  name: string;
  sites: Site[];
  startsAt: string;
  endsAt: string;
  owner: string;
  promotionCount: number;
  status: "planning" | "active" | "ended";
}

export interface AuditEvent {
  id: string;
  timestamp: string;
  actor: string;
  actorRole: Role;
  action: string;
  entityType: string;
  entityId: string;
  entityName: string;
  site: Site;
  before?: string;
  after?: string;
  reason?: string;
  aiAssisted?: boolean;
  automatic?: boolean;
}

export interface User {
  id: string;
  name: string;
  email: string;
  role: Role;
  sites: Site[];
  lastLogin: string;
  status: "active" | "inactive";
}

export interface CatalogItem {
  sku: string;
  site: Site;
  name: string;
  price: number; // cents
  currency: "USD" | "CAD";
  categoryPath: string;
  brand: string;
  dscs: string;
  clearance: boolean;
  promotable: boolean;
}

export interface Scenario {
  id: string;
  name: string;
  templateCode: string;
  site: Site;
  expectedTotal: number;
  lastRunTotal?: number;
  passed?: boolean;
  lastRunAt?: string;
  cart: {
    lines: { sku: string; name: string; qty: number; price: number }[];
    codes?: string[];
    shipping?: { method: string; charge: number };
  };
}

export interface CodePool {
  id: string;
  promotionId: string;
  promotionName: string;
  site: Site;
  type: "public" | "generated" | "imported";
  prefix: string;
  length: number;
  totalCodes: number;
  usedCodes: number;
  createdAt: string;
  codes?: { code: string; used: boolean; usedAt?: string; customerId?: string }[];
}

// ─── Templates ───────────────────────────────────────────────────────────────

export const TEMPLATES: Template[] = [
  { code: "T01", name: "% off individual items", reward: "PERCENT", target: "EACH_ITEM", level: "ITEM" },
  { code: "T02", name: "% off individual items from a category", reward: "PERCENT", target: "EACH_ITEM", level: "ITEM" },
  { code: "T03", name: "% off the subtotal of items", reward: "PERCENT", target: "ITEM_SUBTOTAL", level: "ITEM" },
  { code: "T04", name: "% off the subtotal of items from a category", reward: "PERCENT", target: "ITEM_SUBTOTAL", level: "ITEM" },
  { code: "T05", name: "Amount off individual items", reward: "AMOUNT_OFF", target: "EACH_ITEM", level: "ITEM" },
  { code: "T06", name: "Amount off individual items from a category", reward: "AMOUNT_OFF", target: "EACH_ITEM", level: "ITEM" },
  { code: "T07", name: "Amount off the subtotal of items", reward: "AMOUNT_OFF", target: "ITEM_SUBTOTAL", level: "ITEM" },
  { code: "T08", name: "Amount off the subtotal of items from a category", reward: "AMOUNT_OFF", target: "ITEM_SUBTOTAL", level: "ITEM" },
  { code: "T09", name: "Fixed price on individual items", reward: "FIXED_PRICE", target: "EACH_ITEM", level: "ITEM" },
  { code: "T10", name: "Fixed price on individual items from a category", reward: "FIXED_PRICE", target: "EACH_ITEM", level: "ITEM" },
  { code: "T11", name: "Fixed price on the subtotal of items", reward: "FIXED_PRICE", target: "ITEM_SUBTOTAL", level: "ITEM" },
  { code: "T12", name: "Fixed price on the subtotal of items from a category", reward: "FIXED_PRICE", target: "ITEM_SUBTOTAL", level: "ITEM" },
  { code: "T13", name: "Buy X, get more X at % off", reward: "PERCENT", target: "BUY_GET", level: "ITEM" },
  { code: "T14", name: "Buy from category X, get more from X at % off", reward: "PERCENT", target: "BUY_GET", level: "ITEM" },
  { code: "T15", name: "Buy X, get % off Y", reward: "PERCENT", target: "BUY_GET", level: "ITEM" },
  { code: "T16", name: "Buy from category X, get % off Y", reward: "PERCENT", target: "BUY_GET", level: "ITEM" },
  { code: "T17", name: "Buy A (and B and C), get % off Y", reward: "PERCENT", target: "BUY_GET", level: "ITEM" },
  { code: "T18", name: "Buy A and B, get % off both", reward: "PERCENT", target: "BUY_GET", level: "ITEM" },
  { code: "T19", name: "Buy A (and B and C), get an amount off Y", reward: "AMOUNT_OFF", target: "BUY_GET", level: "ITEM" },
  { code: "T20", name: "Buy X, get Y free", reward: "FREE_ITEM", target: "BUY_GET", level: "ITEM" },
  { code: "T21", name: "Buy from category X, get Y free", reward: "FREE_ITEM", target: "BUY_GET", level: "ITEM" },
  { code: "T22", name: "% off an order", reward: "PERCENT", target: "ORDER", level: "ORDER" },
  { code: "T23", name: "Amount off an order", reward: "AMOUNT_OFF", target: "ORDER", level: "ORDER" },
  { code: "T24", name: "Free gift with an order", reward: "FREE_ITEM", target: "ORDER", level: "ORDER" },
  { code: "T25", name: "Amount off shipping on an order", reward: "AMOUNT_OFF", target: "SHIPPING", level: "SHIPPING" },
  { code: "T26", name: "Fixed price for shipping an order", reward: "FIXED_PRICE", target: "SHIPPING", level: "SHIPPING" },
  { code: "T27", name: "Buy X, get shipping at a fixed price", reward: "FIXED_PRICE", target: "SHIPPING", level: "SHIPPING" },
  { code: "T28", name: "Buy from category X, get shipping at a fixed price", reward: "FIXED_PRICE", target: "SHIPPING", level: "SHIPPING" },
];

// ─── Promotions ──────────────────────────────────────────────────────────────

export const PROMOTIONS: Promotion[] = [
  {
    id: "P-ROD10",
    name: "Rods 10% Off",
    templateCode: "T02",
    templateName: "% off individual items from a category",
    status: "active",
    sites: ["basspro.com", "cabelas.com"],
    campaignId: "C-FALL24",
    campaignName: "Fall Fishing Sale",
    priority: 100,
    version: 3,
    startsAt: "2026-11-01T00:00:00Z",
    endsAt: "2026-12-31T23:59:59Z",
    createdBy: "Maya Ortiz",
    createdAt: "2026-10-15T09:30:00Z",
    updatedAt: "2026-11-10T14:22:00Z",
    reward: { kind: "PERCENT", value: 10, maxDiscount: 5000 },
    conditions: { categories: ["Fishing > Rods"], minQty: 1, channels: ["web", "mobile"] },
    combinationMode: "STACKABLE",
    limits: { totalRedemptions: 10000, perCustomer: 3, usedCount: 4521 },
  },
  {
    id: "P-CLUB5",
    name: "CLUB Members 5% Stacked",
    templateCode: "T22",
    templateName: "% off an order",
    status: "active",
    sites: ["basspro.com", "cabelas.com", "basspro.ca", "mackspw.com"],
    priority: 50,
    version: 1,
    startsAt: "2026-01-01T00:00:00Z",
    endsAt: "2027-12-31T23:59:59Z",
    createdBy: "Rita Gomez",
    createdAt: "2025-12-20T10:00:00Z",
    updatedAt: "2025-12-20T10:00:00Z",
    reward: { kind: "PERCENT", value: 5 },
    conditions: { segments: ["CLUB"], channels: ["web", "mobile", "store"] },
    combinationMode: "STACKABLE",
  },
  {
    id: "P-SHIP99",
    name: "Free Shipping Over $99",
    templateCode: "T25",
    templateName: "Amount off shipping on an order",
    status: "active",
    sites: ["basspro.com"],
    campaignId: "C-FALL24",
    campaignName: "Fall Fishing Sale",
    priority: 200,
    version: 2,
    startsAt: "2026-11-01T00:00:00Z",
    endsAt: "2026-12-31T23:59:59Z",
    createdBy: "Maya Ortiz",
    createdAt: "2026-10-18T11:00:00Z",
    updatedAt: "2026-11-05T09:15:00Z",
    reward: { kind: "AMOUNT_OFF", value: 1200 },
    conditions: { minSpend: 9900, channels: ["web"] },
    combinationMode: "STACKABLE",
  },
  {
    id: "P-BF25",
    name: "Black Friday 25% Off Everything",
    templateCode: "T01",
    templateName: "% off individual items",
    status: "in_review",
    sites: ["basspro.com", "cabelas.com"],
    campaignId: "C-BF26",
    campaignName: "Black Friday 2026",
    priority: 500,
    version: 1,
    startsAt: "2026-11-29T00:00:00Z",
    endsAt: "2026-11-29T23:59:59Z",
    createdBy: "Maya Ortiz",
    createdAt: "2026-11-10T08:00:00Z",
    updatedAt: "2026-11-10T08:00:00Z",
    reward: { kind: "PERCENT", value: 25, maxDiscount: 10000 },
    conditions: { channels: ["web", "mobile", "store"] },
    combinationMode: "EXCLUSIVE",
    limits: { totalRedemptions: 50000, perCustomer: 1, usedCount: 0 },
  },
  {
    id: "P-TAKE20",
    name: "$20 Off $100+ Purchase",
    templateCode: "T23",
    templateName: "Amount off an order",
    status: "active",
    sites: ["basspro.com", "cabelas.com"],
    priority: 150,
    version: 2,
    startsAt: "2026-11-01T00:00:00Z",
    endsAt: "2026-12-15T23:59:59Z",
    createdBy: "Rita Gomez",
    createdAt: "2026-10-25T14:00:00Z",
    updatedAt: "2026-11-01T00:00:00Z",
    reward: { kind: "AMOUNT_OFF", value: 2000 },
    conditions: { minSpend: 10000, channels: ["web", "mobile"] },
    combinationMode: "ONE_PER_LEVEL",
    codes: ["TAKE20"],
    limits: { totalRedemptions: 25000, perCustomer: 2, usedCount: 8734 },
  },
  {
    id: "P-BOGO",
    name: "Buy 2 Lures Get 1 Free",
    templateCode: "T20",
    templateName: "Buy X, get Y free",
    status: "approved",
    sites: ["basspro.com"],
    campaignId: "C-FALL24",
    campaignName: "Fall Fishing Sale",
    priority: 300,
    version: 1,
    startsAt: "2026-12-01T00:00:00Z",
    endsAt: "2026-12-25T23:59:59Z",
    createdBy: "Maya Ortiz",
    createdAt: "2026-11-12T10:30:00Z",
    updatedAt: "2026-11-14T16:00:00Z",
    reward: { kind: "FREE_ITEM", value: 100 },
    conditions: { categories: ["Fishing > Lures"], minQty: 2, channels: ["web", "mobile"] },
    combinationMode: "STACKABLE",
    limits: { totalRedemptions: 5000, perCustomer: 2, usedCount: 0 },
  },
  {
    id: "P-REEL15",
    name: "15% Off Reels",
    templateCode: "T02",
    templateName: "% off individual items from a category",
    status: "draft",
    sites: ["cabelas.com"],
    priority: 120,
    version: 1,
    startsAt: "2026-12-01T00:00:00Z",
    endsAt: "2026-12-31T23:59:59Z",
    createdBy: "Maya Ortiz",
    createdAt: "2026-11-15T09:00:00Z",
    updatedAt: "2026-11-15T09:00:00Z",
    reward: { kind: "PERCENT", value: 15, maxDiscount: 7500 },
    conditions: { categories: ["Fishing > Reels"], channels: ["web"] },
    combinationMode: "STACKABLE",
  },
  {
    id: "P-WELCOME",
    name: "Welcome 10% Off First Order",
    templateCode: "T22",
    templateName: "% off an order",
    status: "active",
    sites: ["basspro.com", "cabelas.com", "basspro.ca"],
    priority: 80,
    version: 1,
    startsAt: "2026-01-01T00:00:00Z",
    endsAt: "2027-12-31T23:59:59Z",
    createdBy: "Rita Gomez",
    createdAt: "2025-12-15T08:00:00Z",
    updatedAt: "2025-12-15T08:00:00Z",
    reward: { kind: "PERCENT", value: 10, maxDiscount: 5000 },
    conditions: { segments: ["new_customer"], channels: ["web", "mobile"] },
    combinationMode: "ONE_PER_LEVEL",
    codes: ["WELCOME"],
    limits: { perCustomer: 1, usedCount: 34210 },
  },
  {
    id: "P-FIXSHIP",
    name: "$4.99 Flat Rate Shipping",
    templateCode: "T26",
    templateName: "Fixed price for shipping an order",
    status: "suspended",
    sites: ["basspro.com"],
    priority: 180,
    version: 3,
    startsAt: "2026-10-01T00:00:00Z",
    endsAt: "2026-11-30T23:59:59Z",
    createdBy: "Maya Ortiz",
    createdAt: "2026-09-20T11:00:00Z",
    updatedAt: "2026-11-08T03:00:00Z",
    reward: { kind: "FIXED_PRICE", value: 499 },
    conditions: { minSpend: 5000, channels: ["web"] },
    combinationMode: "EXCLUSIVE",
    limits: { totalRedemptions: 15000, perCustomer: 5, usedCount: 15000 },
  },
  {
    id: "P-CA10",
    name: "Canada Day 10% Off",
    templateCode: "T01",
    templateName: "% off individual items",
    status: "ended",
    sites: ["basspro.ca"],
    priority: 200,
    version: 2,
    startsAt: "2026-07-01T00:00:00Z",
    endsAt: "2026-07-07T23:59:59Z",
    createdBy: "Rita Gomez",
    createdAt: "2026-06-15T10:00:00Z",
    updatedAt: "2026-07-08T00:00:00Z",
    reward: { kind: "PERCENT", value: 10 },
    conditions: { channels: ["web", "mobile", "store"] },
    combinationMode: "STACKABLE",
    limits: { totalRedemptions: 20000, perCustomer: 3, usedCount: 12450 },
  },
];

// ─── Campaigns ───────────────────────────────────────────────────────────────

export const CAMPAIGNS: Campaign[] = [
  {
    id: "C-FALL24",
    name: "Fall Fishing Sale",
    sites: ["basspro.com", "cabelas.com"],
    startsAt: "2026-11-01",
    endsAt: "2026-12-31",
    owner: "Maya Ortiz",
    promotionCount: 4,
    status: "active",
  },
  {
    id: "C-BF26",
    name: "Black Friday 2026",
    sites: ["basspro.com", "cabelas.com", "mackspw.com"],
    startsAt: "2026-11-29",
    endsAt: "2026-11-29",
    owner: "Rita Gomez",
    promotionCount: 3,
    status: "planning",
  },
  {
    id: "C-CLUB",
    name: "CLUB Days Winter",
    sites: ["basspro.com", "cabelas.com", "basspro.ca", "mackspw.com"],
    startsAt: "2026-12-10",
    endsAt: "2026-12-17",
    owner: "Alan Reed",
    promotionCount: 6,
    status: "planning",
  },
  {
    id: "C-HOLIDAY",
    name: "Holiday Gift Guide",
    sites: ["basspro.com", "cabelas.com"],
    startsAt: "2026-12-01",
    endsAt: "2026-12-25",
    owner: "Maya Ortiz",
    promotionCount: 8,
    status: "planning",
  },
  {
    id: "C-CANADA",
    name: "Canada Day Celebration",
    sites: ["basspro.ca"],
    startsAt: "2026-07-01",
    endsAt: "2026-07-07",
    owner: "Rita Gomez",
    promotionCount: 2,
    status: "ended",
  },
];

// ─── Audit Events ────────────────────────────────────────────────────────────

export const AUDIT_EVENTS: AuditEvent[] = [
  {
    id: "AE-001",
    timestamp: "2026-11-16T14:22:00Z",
    actor: "Maya Ortiz",
    actorRole: "merchandiser",
    action: "promotion.updated",
    entityType: "promotion",
    entityId: "P-ROD10",
    entityName: "Rods 10% Off",
    site: "basspro.com",
    before: "percent: 10, maxDiscount: $40.00",
    after: "percent: 10, maxDiscount: $50.00",
    reason: "Increased max discount cap for holiday season",
  },
  {
    id: "AE-002",
    timestamp: "2026-11-16T13:45:00Z",
    actor: "Dev Patel",
    actorRole: "approver",
    action: "promotion.approved",
    entityType: "promotion",
    entityId: "P-ROD10",
    entityName: "Rods 10% Off",
    site: "basspro.com",
    reason: "Scenarios pass, pricing verified against fall catalog",
  },
  {
    id: "AE-003",
    timestamp: "2026-11-15T09:00:00Z",
    actor: "Maya Ortiz",
    actorRole: "merchandiser",
    action: "promotion.created",
    entityType: "promotion",
    entityId: "P-REEL15",
    entityName: "15% Off Reels",
    site: "cabelas.com",
    aiAssisted: true,
    reason: "AI-assisted draft from brief: 15% off all reels on cabelas.com for December",
  },
  {
    id: "AE-004",
    timestamp: "2026-11-14T16:00:00Z",
    actor: "Dev Patel",
    actorRole: "approver",
    action: "promotion.approved",
    entityType: "promotion",
    entityId: "P-BOGO",
    entityName: "Buy 2 Lures Get 1 Free",
    site: "basspro.com",
    reason: "All 12 scenarios pass, margin impact within budget",
  },
  {
    id: "AE-005",
    timestamp: "2026-11-12T10:30:00Z",
    actor: "Maya Ortiz",
    actorRole: "merchandiser",
    action: "promotion.created",
    entityType: "promotion",
    entityId: "P-BOGO",
    entityName: "Buy 2 Lures Get 1 Free",
    site: "basspro.com",
  },
  {
    id: "AE-006",
    timestamp: "2026-11-10T08:00:00Z",
    actor: "Maya Ortiz",
    actorRole: "merchandiser",
    action: "promotion.submitted",
    entityType: "promotion",
    entityId: "P-BF25",
    entityName: "Black Friday 25% Off Everything",
    site: "basspro.com",
    reason: "Ready for review — all scenarios pass",
  },
  {
    id: "AE-007",
    timestamp: "2026-11-08T03:00:00Z",
    actor: "System",
    actorRole: "super_admin",
    action: "promotion.suspended",
    entityType: "promotion",
    entityId: "P-FIXSHIP",
    entityName: "$4.99 Flat Rate Shipping",
    site: "basspro.com",
    reason: "LIMIT_REACHED — total redemption limit of 15,000 reached",
    automatic: true,
  },
  {
    id: "AE-008",
    timestamp: "2026-11-05T09:15:00Z",
    actor: "Rita Gomez",
    actorRole: "site_admin",
    action: "codes.exported",
    entityType: "code_pool",
    entityId: "CP-TAKE20",
    entityName: "TAKE20 Code Pool",
    site: "basspro.com",
    reason: "Monthly code usage report for finance",
  },
  {
    id: "AE-009",
    timestamp: "2026-11-04T11:30:00Z",
    actor: "Alan Reed",
    actorRole: "super_admin",
    action: "user.role_changed",
    entityType: "user",
    entityId: "U-006",
    entityName: "Quinn Hart",
    site: "basspro.com",
    before: "role: viewer_qa, sites: [basspro.com]",
    after: "role: viewer_qa, sites: [basspro.com, cabelas.com]",
    reason: "Extended QA coverage to cabelas.com for holiday testing",
  },
  {
    id: "AE-010",
    timestamp: "2026-11-01T00:00:01Z",
    actor: "System",
    actorRole: "super_admin",
    action: "promotion.activated",
    entityType: "promotion",
    entityId: "P-TAKE20",
    entityName: "$20 Off $100+ Purchase",
    site: "basspro.com",
    reason: "Scheduled go-live at start date",
    automatic: true,
  },
];

// ─── Users ───────────────────────────────────────────────────────────────────

export const USERS: User[] = [
  {
    id: "U-001",
    name: "Alan Reed",
    email: "alan.reed@basspro.com",
    role: "super_admin",
    sites: ["basspro.com", "cabelas.com", "basspro.ca", "mackspw.com"],
    lastLogin: "2026-11-16T08:00:00Z",
    status: "active",
  },
  {
    id: "U-002",
    name: "Maya Ortiz",
    email: "maya.ortiz@basspro.com",
    role: "merchandiser",
    sites: ["basspro.com", "cabelas.com"],
    lastLogin: "2026-11-16T09:30:00Z",
    status: "active",
  },
  {
    id: "U-003",
    name: "Dev Patel",
    email: "dev.patel@basspro.com",
    role: "approver",
    sites: ["basspro.com", "cabelas.com", "basspro.ca"],
    lastLogin: "2026-11-16T13:00:00Z",
    status: "active",
  },
  {
    id: "U-004",
    name: "Rita Gomez",
    email: "rita.gomez@basspro.com",
    role: "site_admin",
    sites: ["basspro.com", "cabelas.com"],
    lastLogin: "2026-11-16T07:45:00Z",
    status: "active",
  },
  {
    id: "U-005",
    name: "Ingrid Voss",
    email: "ingrid.voss@basspro.com",
    role: "auditor",
    sites: ["basspro.com", "cabelas.com", "basspro.ca", "mackspw.com"],
    lastLogin: "2026-11-15T16:00:00Z",
    status: "active",
  },
  {
    id: "U-006",
    name: "Quinn Hart",
    email: "quinn.hart@basspro.com",
    role: "viewer_qa",
    sites: ["basspro.com", "cabelas.com"],
    lastLogin: "2026-11-16T10:15:00Z",
    status: "active",
  },
  {
    id: "U-007",
    name: "Tom Bell",
    email: "tom.bell@basspro.com",
    role: "site_admin",
    sites: ["basspro.ca"],
    lastLogin: "2026-11-14T14:30:00Z",
    status: "active",
  },
  {
    id: "U-008",
    name: "Sarah Chen",
    email: "sarah.chen@basspro.com",
    role: "merchandiser",
    sites: ["mackspw.com"],
    lastLogin: "2026-11-13T11:00:00Z",
    status: "active",
  },
];

// ─── Catalog ─────────────────────────────────────────────────────────────────

export const CATALOG: CatalogItem[] = [
  { sku: "ROD-1001", site: "basspro.com", name: "Pro Qualifier Spinning Rod 7ft", price: 12999, currency: "USD", categoryPath: "Fishing > Rods > Spinning", brand: "Bass Pro Shops", dscs: "FISH-ROD", clearance: false, promotable: true },
  { sku: "ROD-1002", site: "basspro.com", name: "Tourney Special Casting Rod 6.5ft", price: 8999, currency: "USD", categoryPath: "Fishing > Rods > Casting", brand: "Bass Pro Shops", dscs: "FISH-ROD", clearance: false, promotable: true },
  { sku: "ROD-1003", site: "basspro.com", name: "Spring Fly Rod 9ft 5wt", price: 24999, currency: "USD", categoryPath: "Fishing > Rods > Fly", brand: "Cabela's", dscs: "FISH-ROD", clearance: false, promotable: true },
  { sku: "REEL-2040", site: "basspro.com", name: "Pro Max Spinning Reel Size 30", price: 7999, currency: "USD", categoryPath: "Fishing > Reels > Spinning", brand: "Abu Garcia", dscs: "FISH-REEL", clearance: false, promotable: true },
  { sku: "REEL-2041", site: "basspro.com", name: "Revo X Baitcast Reel", price: 11999, currency: "USD", categoryPath: "Fishing > Reels > Baitcast", brand: "Abu Garcia", dscs: "FISH-REEL", clearance: false, promotable: true },
  { sku: "LURE-0310", site: "basspro.com", name: "XPS Lazer Eye Crankbait", price: 899, currency: "USD", categoryPath: "Fishing > Lures > Crankbaits", brand: "Bass Pro Shops", dscs: "FISH-LURE", clearance: false, promotable: true },
  { sku: "LURE-0311", site: "basspro.com", name: "Torpedo Prop Bait 3.5in", price: 699, currency: "USD", categoryPath: "Fishing > Lures > Topwater", brand: "Heddon", dscs: "FISH-LURE", clearance: false, promotable: true },
  { sku: "LURE-0312", site: "basspro.com", name: "Power Worm 7in Plum (10pk)", price: 399, currency: "USD", categoryPath: "Fishing > Lures > Soft Plastics", brand: "Berkley", dscs: "FISH-LURE", clearance: true, promotable: false },
  { sku: "BOOT-4001", site: "basspro.com", name: "Rubber Hip Waders", price: 4999, currency: "USD", categoryPath: "Clothing > Footwear > Waders", brand: "Bass Pro Shops", dscs: "CLOTH-FOOT", clearance: false, promotable: true },
  { sku: "TENT-5001", site: "basspro.com", name: "Eclipse 4-Person Dome Tent", price: 14999, currency: "USD", categoryPath: "Camping > Tents > Dome", brand: "Bass Pro Shops", dscs: "CAMP-TENT", clearance: false, promotable: true },
  { sku: "KAYAK-6001", site: "basspro.com", name: "Ascend 10T Sit-On-Top Kayak", price: 39999, currency: "USD", categoryPath: "Boating > Kayaks > Sit-On-Top", brand: "Ascend", dscs: "BOAT-KAYAK", clearance: false, promotable: true },
  { sku: "AMMO-7001", site: "basspro.com", name: "Federal 9mm 115gr FMJ (50rd)", price: 1699, currency: "USD", categoryPath: "Shooting > Ammunition > Handgun", brand: "Federal", dscs: "SHOOT-AMMO", clearance: false, promotable: true },
  { sku: "ROD-1001", site: "cabelas.com", name: "Pro Qualifier Spinning Rod 7ft", price: 12999, currency: "USD", categoryPath: "Fishing > Rods > Spinning", brand: "Bass Pro Shops", dscs: "FISH-ROD", clearance: false, promotable: true },
  { sku: "ROD-1001", site: "basspro.ca", name: "Pro Qualifier Spinning Rod 7ft", price: 16999, currency: "CAD", categoryPath: "Fishing > Rods > Spinning", brand: "Bass Pro Shops", dscs: "FISH-ROD", clearance: false, promotable: true },
  { sku: "REEL-2040", site: "cabelas.com", name: "Pro Max Spinning Reel Size 30", price: 7999, currency: "USD", categoryPath: "Fishing > Reels > Spinning", brand: "Abu Garcia", dscs: "FISH-REEL", clearance: false, promotable: true },
];

// ─── Scenarios ───────────────────────────────────────────────────────────────

export const SCENARIOS: Scenario[] = [
  {
    id: "S-01",
    name: "Basic rod discount — single item",
    templateCode: "T02",
    site: "basspro.com",
    expectedTotal: 11699,
    lastRunTotal: 11699,
    passed: true,
    lastRunAt: "2026-11-16T10:00:00Z",
    cart: {
      lines: [{ sku: "ROD-1001", name: "Pro Qualifier Spinning Rod 7ft", qty: 1, price: 12999 }],
      shipping: { method: "standard", charge: 0 },
    },
  },
  {
    id: "S-02",
    name: "Rod + reel with CLUB stacking",
    templateCode: "T02",
    site: "basspro.com",
    expectedTotal: 30674,
    lastRunTotal: 30674,
    passed: true,
    lastRunAt: "2026-11-16T10:00:00Z",
    cart: {
      lines: [
        { sku: "ROD-1001", name: "Pro Qualifier Spinning Rod 7ft", qty: 2, price: 12999 },
        { sku: "REEL-2040", name: "Pro Max Spinning Reel Size 30", qty: 1, price: 7999 },
      ],
      codes: ["TAKE20"],
      shipping: { method: "standard", charge: 1200 },
    },
  },
  {
    id: "S-03",
    name: "Clearance item excluded from discount",
    templateCode: "T02",
    site: "basspro.com",
    expectedTotal: 399,
    lastRunTotal: 399,
    passed: true,
    lastRunAt: "2026-11-16T10:00:00Z",
    cart: {
      lines: [{ sku: "LURE-0312", name: "Power Worm 7in Plum (10pk)", qty: 1, price: 399 }],
    },
  },
  {
    id: "S-04",
    name: "TAKE20 code with $100+ spend",
    templateCode: "T23",
    site: "basspro.com",
    expectedTotal: 10999,
    lastRunTotal: 10999,
    passed: true,
    lastRunAt: "2026-11-16T10:00:00Z",
    cart: {
      lines: [{ sku: "ROD-1001", name: "Pro Qualifier Spinning Rod 7ft", qty: 1, price: 12999 }],
      codes: ["TAKE20"],
    },
  },
  {
    id: "S-05",
    name: "TAKE20 code below $100 threshold",
    templateCode: "T23",
    site: "basspro.com",
    expectedTotal: 7999,
    lastRunTotal: 7999,
    passed: true,
    lastRunAt: "2026-11-16T10:00:00Z",
    cart: {
      lines: [{ sku: "REEL-2040", name: "Pro Max Spinning Reel Size 30", qty: 1, price: 7999 }],
      codes: ["TAKE20"],
    },
  },
  {
    id: "S-06",
    name: "Free shipping over $99",
    templateCode: "T25",
    site: "basspro.com",
    expectedTotal: 12999,
    lastRunTotal: 12999,
    passed: true,
    lastRunAt: "2026-11-16T10:00:00Z",
    cart: {
      lines: [{ sku: "ROD-1001", name: "Pro Qualifier Spinning Rod 7ft", qty: 1, price: 12999 }],
      shipping: { method: "standard", charge: 1200 },
    },
  },
  {
    id: "S-07",
    name: "Buy 2 lures get 1 free",
    templateCode: "T20",
    site: "basspro.com",
    expectedTotal: 1798,
    lastRunTotal: 1798,
    passed: true,
    lastRunAt: "2026-11-16T10:00:00Z",
    cart: {
      lines: [{ sku: "LURE-0310", name: "XPS Lazer Eye Crankbait", qty: 3, price: 899 }],
    },
  },
  {
    id: "S-08",
    name: "Exclusive Black Friday blocks stacking",
    templateCode: "T01",
    site: "basspro.com",
    expectedTotal: 9749,
    lastRunTotal: 9749,
    passed: true,
    lastRunAt: "2026-11-16T10:00:00Z",
    cart: {
      lines: [{ sku: "ROD-1001", name: "Pro Qualifier Spinning Rod 7ft", qty: 1, price: 12999 }],
    },
  },
  {
    id: "S-09",
    name: "Canada site CAD pricing",
    templateCode: "T01",
    site: "basspro.ca",
    expectedTotal: 15299,
    lastRunTotal: 15299,
    passed: true,
    lastRunAt: "2026-11-16T10:00:00Z",
    cart: {
      lines: [{ sku: "ROD-1001", name: "Pro Qualifier Spinning Rod 7ft", qty: 1, price: 16999 }],
    },
  },
  {
    id: "S-10",
    name: "Multi-item order discount allocation",
    templateCode: "T23",
    site: "basspro.com",
    expectedTotal: 31797,
    lastRunTotal: 31897,
    passed: false,
    lastRunAt: "2026-11-16T10:00:00Z",
    cart: {
      lines: [
        { sku: "ROD-1001", name: "Pro Qualifier Spinning Rod 7ft", qty: 2, price: 12999 },
        { sku: "REEL-2040", name: "Pro Max Spinning Reel Size 30", qty: 1, price: 7999 },
      ],
      codes: ["TAKE20"],
      shipping: { method: "standard", charge: 1200 },
    },
  },
];

// ─── Code Pools ──────────────────────────────────────────────────────────────

export const CODE_POOLS: CodePool[] = [
  {
    id: "CP-001",
    promotionId: "P-TAKE20",
    promotionName: "$20 Off $100+ Purchase",
    site: "basspro.com",
    type: "public",
    prefix: "",
    length: 6,
    totalCodes: 1,
    usedCodes: 8734,
    createdAt: "2026-10-25T14:00:00Z",
    codes: [{ code: "TAKE20", used: false }],
  },
  {
    id: "CP-002",
    promotionId: "P-WELCOME",
    promotionName: "Welcome 10% Off First Order",
    site: "basspro.com",
    type: "public",
    prefix: "",
    length: 7,
    totalCodes: 1,
    usedCodes: 34210,
    createdAt: "2025-12-15T08:00:00Z",
    codes: [{ code: "WELCOME", used: false }],
  },
  {
    id: "CP-003",
    promotionId: "P-BF25",
    promotionName: "Black Friday 25% Off Everything",
    site: "basspro.com",
    type: "generated",
    prefix: "BF-",
    length: 10,
    totalCodes: 50000,
    usedCodes: 0,
    createdAt: "2026-11-10T08:30:00Z",
  },
  {
    id: "CP-004",
    promotionId: "P-CLUB5",
    promotionName: "CLUB Members 5% Stacked",
    site: "basspro.com",
    type: "imported",
    prefix: "CLUB-",
    length: 12,
    totalCodes: 100000,
    usedCodes: 45230,
    createdAt: "2025-12-20T10:30:00Z",
  },
];

// ─── Helper Functions ────────────────────────────────────────────────────────

export function formatCents(cents: number, currency: "USD" | "CAD" = "USD"): string {
  const symbol = currency === "CAD" ? "C$" : "$";
  return `${symbol}${(cents / 100).toFixed(2)}`;
}

export function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

export function formatDateTime(iso: string): string {
  return new Date(iso).toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

export function getStatusColor(status: PromotionStatus): string {
  const map: Record<PromotionStatus, string> = {
    draft: "badge-draft",
    in_review: "badge-review",
    approved: "badge-approved",
    active: "badge-active",
    suspended: "badge-suspended",
    ended: "badge-ended",
    archived: "badge-archived",
  };
  return map[status];
}

export function getStatusLabel(status: PromotionStatus): string {
  const map: Record<PromotionStatus, string> = {
    draft: "Draft",
    in_review: "In Review",
    approved: "Approved",
    active: "Active",
    suspended: "Suspended",
    ended: "Ended",
    archived: "Archived",
  };
  return map[status];
}

export function getRoleLabel(role: Role): string {
  const map: Record<Role, string> = {
    merchandiser: "Merchandiser",
    approver: "Approver",
    site_admin: "Site Admin",
    super_admin: "Super Admin",
    viewer_qa: "Viewer/QA",
    auditor: "Auditor",
  };
  return map[role];
}
