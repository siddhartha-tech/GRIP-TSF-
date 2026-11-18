"use client";

import React, { useState } from "react";
import AppShell from "../../../components/AppShell";
import { PageHeader, SiteBadge } from "../../../components/shared";
import { TEMPLATES, type Template } from "../../../data/store";
import {
  ArrowLeft,
  ArrowRight,
  Check,
  ChevronDown,
  Globe,
  Layers,
  Target,
  Combine,
  QrCode,
  Shield,
  Save,
  Sparkles,
} from "lucide-react";
import Link from "next/link";

const STEPS = [
  { id: 1, label: "Template", icon: Layers },
  { id: 2, label: "Scope", icon: Globe },
  { id: 3, label: "Conditions", icon: Target },
  { id: 4, label: "Reward", icon: Target },
  { id: 5, label: "Combines", icon: Combine },
  { id: 6, label: "Codes", icon: QrCode },
  { id: 7, label: "Limits", icon: Shield },
  { id: 8, label: "Review", icon: Check },
];

const SITES = [
  { id: "basspro.com", label: "Bass Pro Shops (US)" },
  { id: "cabelas.com", label: "Cabela's (US)" },
  { id: "basspro.ca", label: "Bass Pro Shops (CA)" },
  { id: "mackspw.com", label: "Mack's Prairie Wings" },
];

const CATEGORIES = [
  "Fishing > Rods > Spinning",
  "Fishing > Rods > Casting",
  "Fishing > Rods > Fly",
  "Fishing > Reels > Spinning",
  "Fishing > Reels > Baitcast",
  "Fishing > Lures > Crankbaits",
  "Fishing > Lures > Topwater",
  "Fishing > Lures > Soft Plastics",
  "Camping > Tents > Dome",
  "Clothing > Footwear > Waders",
  "Boating > Kayaks > Sit-On-Top",
  "Shooting > Ammunition > Handgun",
];

export default function NewPromotionPage() {
  const [currentStep, setCurrentStep] = useState(1);
  const [selectedTemplate, setSelectedTemplate] = useState<Template | null>(null);
  const [formData, setFormData] = useState({
    name: "",
    sites: [] as string[],
    channels: ["web"] as string[],
    startsAt: "",
    endsAt: "",
    categories: [] as string[],
    minQty: "",
    minSpend: "",
    segments: [] as string[],
    rewardValue: "",
    maxDiscount: "",
    combinationMode: "STACKABLE",
    codeType: "none",
    publicCode: "",
    generatePrefix: "BP-",
    generateLength: "8",
    generateQty: "1000",
    totalRedemptions: "",
    perCustomer: "",
    priority: "100",
  });

  const updateForm = (field: string, value: string | string[]) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const toggleSite = (siteId: string) => {
    setFormData((prev) => ({
      ...prev,
      sites: prev.sites.includes(siteId)
        ? prev.sites.filter((s) => s !== siteId)
        : [...prev.sites, siteId],
    }));
  };

  const toggleCategory = (cat: string) => {
    setFormData((prev) => ({
      ...prev,
      categories: prev.categories.includes(cat)
        ? prev.categories.filter((c) => c !== cat)
        : [...prev.categories, cat],
    }));
  };

  const groupedTemplates = {
    "Item Level": TEMPLATES.filter((t) => t.level === "ITEM" && !t.target.includes("BUY")),
    "Buy X Get Y": TEMPLATES.filter((t) => t.target === "BUY_GET"),
    "Order Level": TEMPLATES.filter((t) => t.level === "ORDER"),
    "Shipping Level": TEMPLATES.filter((t) => t.level === "SHIPPING"),
  };

  return (
    <AppShell>
      <div className="p-4 lg:p-6 max-w-[1100px] mx-auto">
        <Link href="/promotions" className="inline-flex items-center gap-1.5 text-sm text-gray-500 hover:text-gray-700 mb-4">
          <ArrowLeft size={14} /> Back to Promotions
        </Link>

        <PageHeader
          title="New Promotion"
          subtitle="Create a promotion using the guided wizard"
        />

        {/* Stepper */}
        <div className="card p-4 mb-6 overflow-x-auto">
          <div className="flex items-center gap-1 min-w-max">
            {STEPS.map((step, idx) => {
              const Icon = step.icon;
              const isActive = currentStep === step.id;
              const isCompleted = currentStep > step.id;
              return (
                <React.Fragment key={step.id}>
                  <button
                    onClick={() => setCurrentStep(step.id)}
                    className={`flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-semibold transition-colors ${
                      isActive
                        ? "bg-bass-green text-white"
                        : isCompleted
                        ? "bg-emerald-50 text-emerald-700"
                        : "bg-gray-50 text-gray-400"
                    }`}
                  >
                    {isCompleted ? (
                      <Check size={14} />
                    ) : (
                      <span className="w-5 h-5 rounded-full border-2 flex items-center justify-center text-[10px] font-bold">
                        {step.id}
                      </span>
                    )}
                    <span className="hidden sm:inline">{step.label}</span>
                  </button>
                  {idx < STEPS.length - 1 && (
                    <div className={`w-6 h-0.5 ${isCompleted ? "bg-emerald-300" : "bg-gray-200"}`} />
                  )}
                </React.Fragment>
              );
            })}
          </div>
        </div>

        {/* Step Content */}
        <div className="card p-6">
          {/* Step 1: Template */}
          {currentStep === 1 && (
            <div>
              <h2 className="text-lg font-bold text-gray-900 mb-1">Choose a Template</h2>
              <p className="text-sm text-gray-500 mb-6">Select from 28 promotion templates grouped by reward target</p>
              {Object.entries(groupedTemplates).map(([group, templates]) => (
                <div key={group} className="mb-6">
                  <h3 className="text-sm font-semibold text-gray-700 mb-3 flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-bass-gold" />
                    {group}
                  </h3>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                    {templates.map((t) => (
                      <button
                        key={t.code}
                        onClick={() => setSelectedTemplate(t)}
                        className={`text-left p-3 rounded-lg border-2 transition-all ${
                          selectedTemplate?.code === t.code
                            ? "border-bass-gold bg-bass-cream/50"
                            : "border-gray-100 hover:border-gray-200"
                        }`}
                      >
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-mono font-bold text-gray-400">{t.code}</span>
                          <span className="text-sm font-medium text-gray-900">{t.name}</span>
                        </div>
                        <div className="flex gap-2 mt-1.5">
                          <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded bg-gray-100 text-gray-500">
                            {t.reward}
                          </span>
                          <span className="text-[10px] font-semibold px-1.5 py-0.5 rounded bg-gray-100 text-gray-500">
                            {t.level}
                          </span>
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Step 2: Scope */}
          {currentStep === 2 && (
            <div>
              <h2 className="text-lg font-bold text-gray-900 mb-1">Who, Where & When</h2>
              <p className="text-sm text-gray-500 mb-6">Define the scope of this promotion</p>
              <div className="space-y-5">
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">Promotion Name</label>
                  <input
                    type="text"
                    value={formData.name}
                    onChange={(e) => updateForm("name", e.target.value)}
                    placeholder="e.g., 10% Off Fishing Rods"
                    className="input-field"
                  />
                </div>
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">Sites</label>
                  <div className="grid grid-cols-2 gap-2">
                    {SITES.map((site) => (
                      <button
                        key={site.id}
                        onClick={() => toggleSite(site.id)}
                        className={`p-3 rounded-lg border-2 text-left transition-all ${
                          formData.sites.includes(site.id)
                            ? "border-bass-gold bg-bass-cream/50"
                            : "border-gray-100 hover:border-gray-200"
                        }`}
                      >
                        <div className="flex items-center gap-2">
                          {formData.sites.includes(site.id) && <Check size={14} className="text-bass-gold" />}
                          <span className="text-sm font-medium">{site.label}</span>
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-semibold text-gray-700 mb-2">Start Date</label>
                    <input
                      type="date"
                      value={formData.startsAt}
                      onChange={(e) => updateForm("startsAt", e.target.value)}
                      className="input-field"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-semibold text-gray-700 mb-2">End Date</label>
                    <input
                      type="date"
                      value={formData.endsAt}
                      onChange={(e) => updateForm("endsAt", e.target.value)}
                      className="input-field"
                    />
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">Priority (0–1000)</label>
                  <input
                    type="number"
                    value={formData.priority}
                    onChange={(e) => updateForm("priority", e.target.value)}
                    min="0"
                    max="1000"
                    className="input-field w-32"
                  />
                  <p className="text-xs text-gray-400 mt-1">Higher priority wins when promotions compete</p>
                </div>
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">Channels</label>
                  <div className="flex gap-2">
                    {["web", "mobile", "store"].map((ch) => (
                      <button
                        key={ch}
                        onClick={() => {
                          const channels = formData.channels.includes(ch)
                            ? formData.channels.filter((c) => c !== ch)
                            : [...formData.channels, ch];
                          updateForm("channels", channels);
                        }}
                        className={`px-4 py-2 rounded-lg border-2 text-sm font-medium capitalize transition-all ${
                          formData.channels.includes(ch)
                            ? "border-bass-gold bg-bass-cream/50 text-bass-gold-dark"
                            : "border-gray-100 text-gray-500 hover:border-gray-200"
                        }`}
                      >
                        {ch}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Step 3: Conditions */}
          {currentStep === 3 && (
            <div>
              <h2 className="text-lg font-bold text-gray-900 mb-1">Conditions (If)</h2>
              <p className="text-sm text-gray-500 mb-6">Define what the cart must contain for this promotion to apply</p>
              <div className="space-y-5">
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">Categories</label>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-h-60 overflow-y-auto">
                    {CATEGORIES.map((cat) => (
                      <button
                        key={cat}
                        onClick={() => toggleCategory(cat)}
                        className={`text-left p-2.5 rounded-lg border text-sm transition-all ${
                          formData.categories.includes(cat)
                            ? "border-bass-gold bg-bass-cream/50 font-medium"
                            : "border-gray-100 hover:border-gray-200"
                        }`}
                      >
                        {formData.categories.includes(cat) && <Check size={12} className="inline mr-1.5 text-bass-gold" />}
                        {cat}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-semibold text-gray-700 mb-2">Minimum Quantity</label>
                    <input
                      type="number"
                      value={formData.minQty}
                      onChange={(e) => updateForm("minQty", e.target.value)}
                      placeholder="e.g., 2"
                      className="input-field"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-semibold text-gray-700 mb-2">Minimum Spend ($)</label>
                    <input
                      type="number"
                      value={formData.minSpend}
                      onChange={(e) => updateForm("minSpend", e.target.value)}
                      placeholder="e.g., 99.00"
                      className="input-field"
                    />
                  </div>
                </div>
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">Customer Segments</label>
                  <div className="flex gap-2">
                    {["CLUB", "new_customer", "loyalty_gold", "employee"].map((seg) => (
                      <button
                        key={seg}
                        onClick={() => {
                          const segments = formData.segments.includes(seg)
                            ? formData.segments.filter((s) => s !== seg)
                            : [...formData.segments, seg];
                          updateForm("segments", segments);
                        }}
                        className={`px-3 py-1.5 rounded-lg border text-xs font-medium transition-all ${
                          formData.segments.includes(seg)
                            ? "border-bass-gold bg-bass-cream/50 text-bass-gold-dark"
                            : "border-gray-200 text-gray-500 hover:border-gray-300"
                        }`}
                      >
                        {seg}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Step 4: Reward */}
          {currentStep === 4 && (
            <div>
              <h2 className="text-lg font-bold text-gray-900 mb-1">Reward (Then)</h2>
              <p className="text-sm text-gray-500 mb-6">Define the discount or reward</p>
              <div className="space-y-5">
                <div className="p-4 rounded-lg bg-gray-50 border border-gray-100">
                  <p className="text-xs text-gray-500 mb-1">Selected Template</p>
                  <p className="text-sm font-semibold">
                    {selectedTemplate ? `${selectedTemplate.code} — ${selectedTemplate.name}` : "No template selected"}
                  </p>
                  <p className="text-xs text-gray-400 mt-1">
                    Reward type: {selectedTemplate?.reward || "—"}
                  </p>
                </div>
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">
                    {selectedTemplate?.reward === "PERCENT" ? "Percentage Off" :
                     selectedTemplate?.reward === "AMOUNT_OFF" ? "Amount Off (cents)" :
                     selectedTemplate?.reward === "FIXED_PRICE" ? "Fixed Price (cents)" :
                     "Reward Value"}
                  </label>
                  <input
                    type="number"
                    value={formData.rewardValue}
                    onChange={(e) => updateForm("rewardValue", e.target.value)}
                    placeholder={selectedTemplate?.reward === "PERCENT" ? "e.g., 10" : "e.g., 2000"}
                    className="input-field w-48"
                  />
                  {selectedTemplate?.reward === "PERCENT" && (
                    <p className="text-xs text-gray-400 mt-1">Enter a value between 1 and 100</p>
                  )}
                </div>
                {(selectedTemplate?.reward === "PERCENT" || selectedTemplate?.reward === "AMOUNT_OFF") && (
                  <div>
                    <label className="block text-sm font-semibold text-gray-700 mb-2">Max Discount (cents, optional)</label>
                    <input
                      type="number"
                      value={formData.maxDiscount}
                      onChange={(e) => updateForm("maxDiscount", e.target.value)}
                      placeholder="e.g., 5000 ($50.00)"
                      className="input-field w-48"
                    />
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Step 5: Combines */}
          {currentStep === 5 && (
            <div>
              <h2 className="text-lg font-bold text-gray-900 mb-1">Combination Mode</h2>
              <p className="text-sm text-gray-500 mb-6">How this promotion interacts with others</p>
              <div className="space-y-3">
                {[
                  { value: "STACKABLE", label: "Stackable", desc: "Can be combined with other stackable promotions" },
                  { value: "EXCLUSIVE", label: "Exclusive", desc: "Only this promotion applies — blocks all others on the order" },
                  { value: "BEST_DEAL", label: "Best Deal", desc: "System picks whichever gives the customer the best price" },
                  { value: "ONE_PER_LEVEL", label: "One Per Level", desc: "Only one promotion per level (item, order, shipping)" },
                ].map((mode) => (
                  <button
                    key={mode.value}
                    onClick={() => updateForm("combinationMode", mode.value)}
                    className={`w-full text-left p-4 rounded-lg border-2 transition-all ${
                      formData.combinationMode === mode.value
                        ? "border-bass-gold bg-bass-cream/50"
                        : "border-gray-100 hover:border-gray-200"
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <div className={`w-4 h-4 rounded-full border-2 flex items-center justify-center ${
                        formData.combinationMode === mode.value ? "border-bass-gold" : "border-gray-300"
                      }`}>
                        {formData.combinationMode === mode.value && (
                          <div className="w-2 h-2 rounded-full bg-bass-gold" />
                        )}
                      </div>
                      <span className="text-sm font-semibold text-gray-900">{mode.label}</span>
                    </div>
                    <p className="text-xs text-gray-500 mt-1 ml-6">{mode.desc}</p>
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Step 6: Codes */}
          {currentStep === 6 && (
            <div>
              <h2 className="text-lg font-bold text-gray-900 mb-1">Promotion Codes</h2>
              <p className="text-sm text-gray-500 mb-6">Optionally require a code for redemption</p>
              <div className="space-y-4">
                {[
                  { value: "none", label: "No Code Required", desc: "Promotion applies automatically" },
                  { value: "public", label: "Public Code", desc: "A single shared code (e.g., TAKE20)" },
                  { value: "generated", label: "Generated Codes", desc: "Unique codes generated in bulk" },
                ].map((opt) => (
                  <button
                    key={opt.value}
                    onClick={() => updateForm("codeType", opt.value)}
                    className={`w-full text-left p-4 rounded-lg border-2 transition-all ${
                      formData.codeType === opt.value
                        ? "border-bass-gold bg-bass-cream/50"
                        : "border-gray-100 hover:border-gray-200"
                    }`}
                  >
                    <span className="text-sm font-semibold text-gray-900">{opt.label}</span>
                    <p className="text-xs text-gray-500 mt-0.5">{opt.desc}</p>
                  </button>
                ))}
                {formData.codeType === "public" && (
                  <div className="ml-4 p-4 bg-gray-50 rounded-lg">
                    <label className="block text-sm font-semibold text-gray-700 mb-2">Code</label>
                    <input
                      type="text"
                      value={formData.publicCode}
                      onChange={(e) => updateForm("publicCode", e.target.value.toUpperCase())}
                      placeholder="e.g., TAKE20"
                      className="input-field w-48 font-mono uppercase"
                    />
                  </div>
                )}
                {formData.codeType === "generated" && (
                  <div className="ml-4 p-4 bg-gray-50 rounded-lg space-y-3">
                    <div className="grid grid-cols-3 gap-3">
                      <div>
                        <label className="block text-xs font-semibold text-gray-700 mb-1">Prefix</label>
                        <input
                          type="text"
                          value={formData.generatePrefix}
                          onChange={(e) => updateForm("generatePrefix", e.target.value.toUpperCase())}
                          className="input-field font-mono"
                        />
                      </div>
                      <div>
                        <label className="block text-xs font-semibold text-gray-700 mb-1">Length (6–16)</label>
                        <input
                          type="number"
                          value={formData.generateLength}
                          onChange={(e) => updateForm("generateLength", e.target.value)}
                          min="6"
                          max="16"
                          className="input-field"
                        />
                      </div>
                      <div>
                        <label className="block text-xs font-semibold text-gray-700 mb-1">Quantity</label>
                        <input
                          type="number"
                          value={formData.generateQty}
                          onChange={(e) => updateForm("generateQty", e.target.value)}
                          className="input-field"
                        />
                      </div>
                    </div>
                    <button className="btn-secondary text-xs">Generate Preview</button>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Step 7: Limits */}
          {currentStep === 7 && (
            <div>
              <h2 className="text-lg font-bold text-gray-900 mb-1">Redemption Limits</h2>
              <p className="text-sm text-gray-500 mb-6">Control how many times this promotion can be used</p>
              <div className="space-y-5">
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">Total Redemptions (optional)</label>
                  <input
                    type="number"
                    value={formData.totalRedemptions}
                    onChange={(e) => updateForm("totalRedemptions", e.target.value)}
                    placeholder="e.g., 10000"
                    className="input-field w-48"
                  />
                  <p className="text-xs text-gray-400 mt-1">Promotion will be automatically suspended when this limit is reached</p>
                </div>
                <div>
                  <label className="block text-sm font-semibold text-gray-700 mb-2">Per Customer (optional)</label>
                  <input
                    type="number"
                    value={formData.perCustomer}
                    onChange={(e) => updateForm("perCustomer", e.target.value)}
                    placeholder="e.g., 3"
                    className="input-field w-48"
                  />
                </div>
              </div>
            </div>
          )}

          {/* Step 8: Review */}
          {currentStep === 8 && (
            <div>
              <h2 className="text-lg font-bold text-gray-900 mb-1">Review & Save</h2>
              <p className="text-sm text-gray-500 mb-6">Review your promotion before saving as a draft</p>
              <div className="space-y-4">
                <div className="p-4 rounded-lg bg-gray-50 border border-gray-100">
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <p className="text-xs text-gray-500">Template</p>
                      <p className="font-semibold">{selectedTemplate?.code} — {selectedTemplate?.name || "Not selected"}</p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-500">Name</p>
                      <p className="font-semibold">{formData.name || "—"}</p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-500">Sites</p>
                      <div className="flex gap-1 mt-1">
                        {formData.sites.length > 0 ? formData.sites.map((s) => <SiteBadge key={s} site={s} />) : <span className="text-gray-400">None</span>}
                      </div>
                    </div>
                    <div>
                      <p className="text-xs text-gray-500">Priority</p>
                      <p className="font-semibold font-mono">{formData.priority}</p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-500">Reward</p>
                      <p className="font-semibold">{formData.rewardValue || "—"}{selectedTemplate?.reward === "PERCENT" ? "%" : " cents"}</p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-500">Combination</p>
                      <p className="font-semibold">{formData.combinationMode}</p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-500">Schedule</p>
                      <p className="font-semibold">{formData.startsAt || "—"} → {formData.endsAt || "—"}</p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-500">Codes</p>
                      <p className="font-semibold capitalize">{formData.codeType === "none" ? "No code" : formData.codeType === "public" ? formData.publicCode || "Public" : "Generated"}</p>
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-3 p-4 rounded-lg bg-amber-50 border border-amber-200">
                  <Sparkles size={16} className="text-amber-500" />
                  <p className="text-sm text-amber-800">
                    This promotion will be saved as a <strong>Draft</strong>. Submit it for review when ready.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Navigation */}
          <div className="flex items-center justify-between mt-8 pt-6 border-t border-gray-100">
            <button
              onClick={() => setCurrentStep(Math.max(1, currentStep - 1))}
              disabled={currentStep === 1}
              className="btn-secondary disabled:opacity-40"
            >
              <ArrowLeft size={14} /> Previous
            </button>
            {currentStep < 8 ? (
              <button
                onClick={() => setCurrentStep(Math.min(8, currentStep + 1))}
                className="btn-primary"
              >
                Next <ArrowRight size={14} />
              </button>
            ) : (
              <button className="btn-gold">
                <Save size={14} /> Save as Draft
              </button>
            )}
          </div>
        </div>
      </div>
    </AppShell>
  );
}
