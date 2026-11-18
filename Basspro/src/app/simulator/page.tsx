"use client";

import React, { useState } from "react";
import AppShell from "../../components/AppShell";
import { PageHeader, SiteBadge, MoneyText } from "../../components/shared";
import { CATALOG, PROMOTIONS, formatCents } from "../../data/store";
import {
  Play,
  Plus,
  Minus,
  Trash2,
  CheckCircle2,
  XCircle,
  ArrowRight,
  ShoppingCart,
  Truck,
  CreditCard,
  Tag,
  Search,
} from "lucide-react";

interface CartLine {
  sku: string;
  name: string;
  qty: number;
  price: number;
}

interface SimResult {
  merchandise: number;
  shipping: number;
  discounts: number;
  total: number;
  applied: { name: string; amount: number; promoId: string }[];
  notApplied: { name: string; reason: string; promoId: string }[];
  trace: string[];
}

export default function SimulatorPage() {
  const [cart, setCart] = useState<CartLine[]>([
    { sku: "ROD-1001", name: "Pro Qualifier Spinning Rod 7ft", qty: 2, price: 12999 },
    { sku: "REEL-2040", name: "Pro Max Spinning Reel Size 30", qty: 1, price: 7999 },
    { sku: "LURE-0310", name: "XPS Lazer Eye Crankbait", qty: 3, price: 899 },
  ]);
  const [site, setSite] = useState("basspro.com");
  const [channel, setChannel] = useState("web");
  const [segment, setSegment] = useState("CLUB");
  const [codes, setCodes] = useState("TAKE20");
  const [shippingMethod, setShippingMethod] = useState("standard");
  const [shippingCharge, setShippingCharge] = useState(1200);
  const [includeDraft, setIncludeDraft] = useState("");
  const [hasRun, setHasRun] = useState(false);
  const [searchCatalog, setSearchCatalog] = useState("");

  const merchandise = cart.reduce((sum, l) => sum + l.price * l.qty, 0);

  const dummyResult: SimResult = {
    merchandise: 36994,
    shipping: 1200,
    discounts: 7520,
    total: 30674,
    applied: [
      { name: "Rods 10% Off", amount: 2600, promoId: "P-ROD10" },
      { name: "CLUB Members 5% Stacked", amount: 1170, promoId: "P-CLUB5" },
      { name: "$20 Off $100+ Purchase", amount: 2000, promoId: "P-TAKE20" },
      { name: "Free Shipping Over $99", amount: 1200, promoId: "P-SHIP99" },
      { name: "Buy 2 Lures Get 1 Free", amount: 550, promoId: "P-BOGO" },
    ],
    notApplied: [
      { name: "Welcome 10% Off First Order", reason: "SEGMENT_NOT_MATCHED", promoId: "P-WELCOME" },
      { name: "$4.99 Flat Rate Shipping", reason: "LIMIT_REACHED", promoId: "P-FIXSHIP" },
    ],
    trace: [
      "1. Find: 7 promotions live for basspro.com",
      "2. Qualify: 5 promotions match cart conditions",
      "3. Rank: P-ROD10 (pri:100), P-CLUB5 (pri:50), P-TAKE20 (pri:150)",
      "4. Combine: STACKABLE mode — all 3 item/order promos applied",
      "5. Calculate: Item discounts: $26.00 + $5.50, Order: $20.00 + $11.70, Shipping: $12.00",
      "6. Integrity: ✓ Line totals sum to order total",
      "7. Integrity: ✓ No line price below zero",
      "8. Integrity: ✓ Same hash on re-evaluation",
      "9. Integrity: ✓ Allocation cents sum to discount",
      "10. Integrity: ✓ Deterministic result",
    ],
  };

  const filteredCatalog = CATALOG.filter(
    (item) =>
      item.site === site &&
      (searchCatalog === "" ||
        item.name.toLowerCase().includes(searchCatalog.toLowerCase()) ||
        item.sku.toLowerCase().includes(searchCatalog.toLowerCase()))
  );

  const addToCart = (item: typeof CATALOG[0]) => {
    const existing = cart.find((l) => l.sku === item.sku);
    if (existing) {
      setCart(cart.map((l) => (l.sku === item.sku ? { ...l, qty: l.qty + 1 } : l)));
    } else {
      setCart([...cart, { sku: item.sku, name: item.name, qty: 1, price: item.price }]);
    }
  };

  return (
    <AppShell>
      <div className="p-4 lg:p-6 max-w-[1400px] mx-auto">
        <PageHeader
          title="Cart Simulator"
          subtitle="Test cart pricing with the same evaluator as checkout"
          actions={
            <button
              onClick={() => setHasRun(true)}
              className="btn-primary"
            >
              <Play size={16} /> Run Simulation
            </button>
          }
        />

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Cart Builder */}
          <div className="space-y-4">
            {/* Site & Context */}
            <div className="card p-5">
              <h3 className="text-sm font-semibold text-gray-900 mb-3">Context</h3>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-semibold text-gray-500 mb-1">Site</label>
                  <select
                    value={site}
                    onChange={(e) => setSite(e.target.value)}
                    className="input-field"
                  >
                    <option value="basspro.com">basspro.com</option>
                    <option value="cabelas.com">cabelas.com</option>
                    <option value="basspro.ca">basspro.ca</option>
                    <option value="mackspw.com">mackspw.com</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-semibold text-gray-500 mb-1">Channel</label>
                  <select value={channel} onChange={(e) => setChannel(e.target.value)} className="input-field">
                    <option value="web">Web</option>
                    <option value="mobile">Mobile</option>
                    <option value="store">Store</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-semibold text-gray-500 mb-1">Customer Segment</label>
                  <select value={segment} onChange={(e) => setSegment(e.target.value)} className="input-field">
                    <option value="">None</option>
                    <option value="CLUB">CLUB Member</option>
                    <option value="new_customer">New Customer</option>
                    <option value="loyalty_gold">Loyalty Gold</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-semibold text-gray-500 mb-1">Promo Code</label>
                  <input
                    type="text"
                    value={codes}
                    onChange={(e) => setCodes(e.target.value.toUpperCase())}
                    placeholder="e.g., TAKE20"
                    className="input-field font-mono"
                  />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3 mt-3">
                <div>
                  <label className="block text-xs font-semibold text-gray-500 mb-1">Shipping Method</label>
                  <select value={shippingMethod} onChange={(e) => setShippingMethod(e.target.value)} className="input-field">
                    <option value="standard">Standard</option>
                    <option value="express">Express</option>
                    <option value="pickup">Store Pickup</option>
                    <option value="freight">Freight</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-semibold text-gray-500 mb-1">Shipping Charge (¢)</label>
                  <input
                    type="number"
                    value={shippingCharge}
                    onChange={(e) => setShippingCharge(Number(e.target.value))}
                    className="input-field"
                  />
                </div>
              </div>
              <div className="mt-3">
                <label className="block text-xs font-semibold text-gray-500 mb-1">Include Draft Promotion</label>
                <select value={includeDraft} onChange={(e) => setIncludeDraft(e.target.value)} className="input-field">
                  <option value="">None (live only)</option>
                  {PROMOTIONS.filter((p) => p.status === "draft" || p.status === "in_review").map((p) => (
                    <option key={p.id} value={p.id}>{p.name} ({p.id})</option>
                  ))}
                </select>
              </div>
            </div>

            {/* Cart Lines */}
            <div className="card p-5">
              <h3 className="text-sm font-semibold text-gray-900 mb-3 flex items-center gap-2">
                <ShoppingCart size={16} className="text-gray-400" /> Cart Lines
              </h3>
              <div className="space-y-2 mb-4">
                {cart.map((line, idx) => (
                  <div key={line.sku} className="flex items-center gap-3 p-3 rounded-lg bg-gray-50 border border-gray-100">
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-gray-900 truncate">{line.name}</p>
                      <p className="text-xs text-gray-400 font-mono">{line.sku} · {formatCents(line.price)}</p>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <button
                        onClick={() => setCart(cart.map((l, i) => i === idx ? { ...l, qty: Math.max(1, l.qty - 1) } : l))}
                        className="p-1 rounded bg-white border border-gray-200 hover:bg-gray-50"
                      >
                        <Minus size={12} />
                      </button>
                      <span className="w-8 text-center text-sm font-bold">{line.qty}</span>
                      <button
                        onClick={() => setCart(cart.map((l, i) => i === idx ? { ...l, qty: l.qty + 1 } : l))}
                        className="p-1 rounded bg-white border border-gray-200 hover:bg-gray-50"
                      >
                        <Plus size={12} />
                      </button>
                    </div>
                    <span className="text-sm font-bold w-20 text-right">{formatCents(line.price * line.qty)}</span>
                    <button
                      onClick={() => setCart(cart.filter((_, i) => i !== idx))}
                      className="p-1 text-gray-400 hover:text-red-500"
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                ))}
              </div>

              {/* Add from catalog */}
              <div className="border-t border-gray-100 pt-3">
                <div className="relative mb-2">
                  <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
                  <input
                    type="text"
                    value={searchCatalog}
                    onChange={(e) => setSearchCatalog(e.target.value)}
                    placeholder="Search catalog to add items..."
                    className="input-field pl-8 text-xs"
                  />
                </div>
                {searchCatalog && (
                  <div className="max-h-40 overflow-y-auto space-y-1">
                    {filteredCatalog.slice(0, 6).map((item) => (
                      <button
                        key={item.sku}
                        onClick={() => { addToCart(item); setSearchCatalog(""); }}
                        className="w-full flex items-center justify-between p-2 rounded text-xs hover:bg-gray-50"
                      >
                        <span className="font-medium">{item.name}</span>
                        <span className="text-gray-500 font-mono">{formatCents(item.price)}</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>

              <div className="flex justify-between items-center mt-4 pt-3 border-t border-gray-100">
                <span className="text-sm text-gray-500">Merchandise Subtotal</span>
                <span className="text-lg font-bold">{formatCents(merchandise)}</span>
              </div>
            </div>
          </div>

          {/* Results */}
          <div className="space-y-4">
            {hasRun ? (
              <>
                {/* Summary */}
                <div className="card p-5">
                  <h3 className="text-sm font-semibold text-gray-900 mb-4">Pricing Result</h3>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-500">Merchandise</span>
                      <span className="font-semibold">{formatCents(dummyResult.merchandise)}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-500">Shipping</span>
                      <span className="font-semibold">{formatCents(dummyResult.shipping)}</span>
                    </div>
                    <div className="flex justify-between text-sm text-emerald-600">
                      <span>Discounts</span>
                      <span className="font-bold">−{formatCents(dummyResult.discounts)}</span>
                    </div>
                    <div className="flex justify-between text-lg pt-2 border-t border-gray-200">
                      <span className="font-bold">Total Before Tax</span>
                      <span className="font-bold text-bass-green">{formatCents(dummyResult.total)}</span>
                    </div>
                  </div>
                </div>

                {/* Applied */}
                <div className="card p-5">
                  <h3 className="text-sm font-semibold text-gray-900 mb-3 flex items-center gap-2">
                    <CheckCircle2 size={16} className="text-emerald-500" /> Applied Promotions
                  </h3>
                  <div className="space-y-2">
                    {dummyResult.applied.map((a) => (
                      <div key={a.promoId} className="flex items-center justify-between p-3 rounded-lg bg-emerald-50 border border-emerald-100">
                        <div className="flex items-center gap-2">
                          <CheckCircle2 size={14} className="text-emerald-500" />
                          <div>
                            <span className="text-sm font-medium text-emerald-800">{a.name}</span>
                            <span className="text-xs text-emerald-600 ml-2">{a.promoId}</span>
                          </div>
                        </div>
                        <span className="text-sm font-bold text-emerald-700">−{formatCents(a.amount)}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Not Applied */}
                <div className="card p-5">
                  <h3 className="text-sm font-semibold text-gray-900 mb-3 flex items-center gap-2">
                    <XCircle size={16} className="text-gray-400" /> Not Applied
                  </h3>
                  <div className="space-y-2">
                    {dummyResult.notApplied.map((na) => (
                      <div key={na.promoId} className="flex items-center justify-between p-3 rounded-lg bg-gray-50 border border-gray-100">
                        <div className="flex items-center gap-2">
                          <XCircle size={14} className="text-gray-400" />
                          <span className="text-sm text-gray-600">{na.name}</span>
                        </div>
                        <span className="text-xs font-mono text-gray-400 bg-gray-100 px-2 py-0.5 rounded">{na.reason}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Trace */}
                <div className="card p-5">
                  <h3 className="text-sm font-semibold text-gray-900 mb-3">Evaluation Trace</h3>
                  <div className="space-y-1.5 font-mono text-xs">
                    {dummyResult.trace.map((step, idx) => (
                      <div key={idx} className={`p-2 rounded ${step.includes("✓") ? "bg-emerald-50 text-emerald-700" : "bg-gray-50 text-gray-600"}`}>
                        {step}
                      </div>
                    ))}
                  </div>
                </div>

                {/* Live vs Draft */}
                {includeDraft && (
                  <div className="card p-5">
                    <h3 className="text-sm font-semibold text-gray-900 mb-3">Live vs Draft Comparison</h3>
                    <div className="grid grid-cols-3 gap-3">
                      <div className="p-3 rounded-lg bg-gray-50 text-center">
                        <p className="text-xs text-gray-500 mb-1">Live</p>
                        <p className="text-lg font-bold">$318.44</p>
                      </div>
                      <div className="p-3 rounded-lg bg-emerald-50 text-center">
                        <p className="text-xs text-emerald-600 mb-1">With Draft</p>
                        <p className="text-lg font-bold text-emerald-700">$306.74</p>
                      </div>
                      <div className="p-3 rounded-lg bg-blue-50 text-center">
                        <p className="text-xs text-blue-600 mb-1">Difference</p>
                        <p className="text-lg font-bold text-blue-700">−$11.70</p>
                      </div>
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="card p-16 text-center">
                <Play size={40} className="text-gray-300 mx-auto mb-3" />
                <p className="text-lg font-semibold text-gray-400">Run a simulation</p>
                <p className="text-sm text-gray-300 mt-1">Build your cart and click "Run Simulation" to see pricing results</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </AppShell>
  );
}
