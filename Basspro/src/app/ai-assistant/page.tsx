"use client";

import React, { useState } from "react";
import AppShell from "../../components/AppShell";
import { PageHeader, SiteBadge } from "../../components/shared";
import {
  Sparkles,
  Send,
  Wand2,
  CheckCircle2,
  AlertTriangle,
  Edit,
  Eye,
  ArrowRight,
  MessageSquare,
  Lightbulb,
} from "lucide-react";

export default function AIAssistantPage() {
  const [brief, setBrief] = useState("");
  const [hasResult, setHasResult] = useState(false);
  const [clarifying, setClarifying] = useState(false);

  const handleSubmit = () => {
    if (brief.trim()) {
      setClarifying(true);
      setTimeout(() => {
        setClarifying(false);
        setHasResult(true);
      }, 800);
    }
  };

  return (
    <AppShell>
      <div className="p-4 lg:p-6 max-w-[1100px] mx-auto">
        <PageHeader
          title="AI Assistant"
          subtitle="Describe a promotion in plain words and get a structured draft"
        />

        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
          {/* Input */}
          <div className="lg:col-span-2 space-y-4">
            <div className="card p-5">
              <h3 className="text-sm font-semibold text-gray-900 mb-3 flex items-center gap-2">
                <MessageSquare size={16} className="text-purple-500" /> Your Brief
              </h3>
              <textarea
                value={brief}
                onChange={(e) => setBrief(e.target.value)}
                placeholder="Describe the promotion you want to create...&#10;&#10;Example: 10% off rods when you buy 2+, not clearance, basspro.com from 1 Dec"
                className="input-field h-40 resize-none mb-3"
              />
              <button
                onClick={handleSubmit}
                className="btn-gold w-full"
                disabled={!brief.trim()}
              >
                <Sparkles size={14} /> Draft with AI
              </button>
            </div>

            {/* Example briefs */}
            <div className="card p-5">
              <h3 className="text-xs font-semibold text-gray-500 mb-3 flex items-center gap-2">
                <Lightbulb size={14} /> Example Briefs
              </h3>
              <div className="space-y-2">
                {[
                  "10% off rods when you buy 2+, not clearance, basspro.com from 1 Dec",
                  "Free shipping on orders over $99 for CLUB members, all US sites",
                  "$20 off orders over $100 with code TAKE20, basspro.com and cabelas.com",
                  "Buy 2 lures get 1 free, basspro.com only, December 1-25",
                  "25% off everything for Black Friday, exclusive, all sites, one day only",
                ].map((example, idx) => (
                  <button
                    key={idx}
                    onClick={() => setBrief(example)}
                    className="w-full text-left p-2.5 rounded-lg border border-gray-100 hover:border-purple-200 hover:bg-purple-50/30 transition-colors text-xs text-gray-600"
                  >
                    &ldquo;{example}&rdquo;
                  </button>
                ))}
              </div>
            </div>

            {/* Constraints */}
            <div className="card p-4 bg-amber-50 border-amber-200">
              <div className="flex items-start gap-2">
                <AlertTriangle size={14} className="text-amber-500 mt-0.5 flex-shrink-0" />
                <div className="text-xs text-amber-700">
                  <p className="font-semibold mb-1">AI Assistant Constraints</p>
                  <ul className="space-y-0.5 list-disc list-inside">
                    <li>Cannot submit, approve, or activate promotions</li>
                    <li>Sees only your sites&apos; catalog and templates</li>
                    <li>No customer data access</li>
                    <li>AI-filled fields are marked for review</li>
                    <li>Drafts flagged as AI-assisted in audit trail</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>

          {/* Result */}
          <div className="lg:col-span-3 space-y-4">
            {clarifying && (
              <div className="card p-5">
                <h3 className="text-sm font-semibold text-gray-900 mb-3 flex items-center gap-2">
                  <Sparkles size={16} className="text-purple-500 animate-pulse" /> Clarifying...
                </h3>
                <div className="p-4 rounded-lg bg-purple-50 border border-purple-200">
                  <p className="text-sm text-purple-800 font-medium mb-3">Which rods did you mean?</p>
                  <div className="flex flex-wrap gap-2">
                    {["All rods", "Spinning", "Casting", "Fly"].map((opt) => (
                      <button
                        key={opt}
                        className="px-3 py-1.5 rounded-full border border-purple-300 text-xs font-medium text-purple-700 hover:bg-purple-100 transition-colors"
                      >
                        {opt}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {hasResult && (
              <>
                {/* Read-back */}
                <div className="card p-5">
                  <div className="flex items-center gap-2 mb-3">
                    <Sparkles size={16} className="text-purple-500" />
                    <h3 className="text-sm font-semibold text-gray-900">AI Read-Back</h3>
                    <span className="badge bg-purple-100 text-purple-700 text-[10px]">AI-filled</span>
                  </div>
                  <div className="p-4 rounded-lg bg-purple-50 border border-purple-200">
                    <p className="text-sm text-purple-900">
                      <strong>10% off each rod</strong> in <em>Fishing › Rods</em> when <strong>2+ bought</strong>,
                      excluding clearance items, on <strong>basspro.com</strong>, web channel,
                      from <strong>Dec 1, 2026</strong> to <strong>Dec 31, 2026</strong>.
                      Stackable with other promotions. No code required.
                    </p>
                  </div>
                </div>

                {/* Generated Draft */}
                <div className="card p-5">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-sm font-semibold text-gray-900 flex items-center gap-2">
                      <Wand2 size={16} className="text-bass-gold" /> Generated Draft
                    </h3>
                    <div className="flex gap-2">
                      <button className="btn-secondary text-xs py-1.5">
                        <Edit size={12} /> Edit in Wizard
                      </button>
                      <button className="btn-primary text-xs py-1.5">
                        <CheckCircle2 size={12} /> Accept Draft
                      </button>
                    </div>
                  </div>

                  <div className="space-y-4">
                    {/* Template */}
                    <div className="p-3 rounded-lg bg-gray-50 border border-gray-100">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-[10px] font-bold text-purple-600 bg-purple-100 px-1.5 py-0.5 rounded">AI</span>
                        <span className="text-xs text-gray-500">Template</span>
                      </div>
                      <p className="text-sm font-semibold">T02 — % off individual items from a category</p>
                    </div>

                    {/* Scope */}
                    <div className="p-3 rounded-lg bg-gray-50 border border-gray-100">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-[10px] font-bold text-purple-600 bg-purple-100 px-1.5 py-0.5 rounded">AI</span>
                        <span className="text-xs text-gray-500">Scope (Who / Where / When)</span>
                      </div>
                      <div className="grid grid-cols-2 gap-3 text-sm">
                        <div>
                          <p className="text-xs text-gray-400">Name</p>
                          <p className="font-medium">10% Off Rods (2+ Items)</p>
                        </div>
                        <div>
                          <p className="text-xs text-gray-400">Sites</p>
                          <SiteBadge site="basspro.com" />
                        </div>
                        <div>
                          <p className="text-xs text-gray-400">Start</p>
                          <p className="font-medium">Dec 1, 2026</p>
                        </div>
                        <div>
                          <p className="text-xs text-gray-400">End</p>
                          <p className="font-medium">Dec 31, 2026</p>
                        </div>
                        <div>
                          <p className="text-xs text-gray-400">Channel</p>
                          <p className="font-medium">Web</p>
                        </div>
                        <div>
                          <p className="text-xs text-gray-400">Priority</p>
                          <p className="font-medium font-mono">100</p>
                        </div>
                      </div>
                    </div>

                    {/* Conditions */}
                    <div className="p-3 rounded-lg bg-gray-50 border border-gray-100">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-[10px] font-bold text-purple-600 bg-purple-100 px-1.5 py-0.5 rounded">AI</span>
                        <span className="text-xs text-gray-500">Conditions (If)</span>
                      </div>
                      <div className="grid grid-cols-2 gap-3 text-sm">
                        <div>
                          <p className="text-xs text-gray-400">Category</p>
                          <span className="text-xs bg-white border border-gray-200 px-2 py-0.5 rounded">Fishing &gt; Rods</span>
                        </div>
                        <div>
                          <p className="text-xs text-gray-400">Min Quantity</p>
                          <p className="font-medium">2+</p>
                        </div>
                        <div>
                          <p className="text-xs text-gray-400">Exclusions</p>
                          <span className="text-xs bg-red-50 text-red-600 border border-red-200 px-2 py-0.5 rounded">Clearance items</span>
                        </div>
                      </div>
                    </div>

                    {/* Reward */}
                    <div className="p-3 rounded-lg bg-bass-cream/50 border border-bass-gold/20">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-[10px] font-bold text-purple-600 bg-purple-100 px-1.5 py-0.5 rounded">AI</span>
                        <span className="text-xs text-gray-500">Reward (Then)</span>
                      </div>
                      <p className="text-lg font-bold text-gray-900">10% Off</p>
                      <p className="text-xs text-gray-500">Per qualifying item</p>
                    </div>

                    {/* Combination */}
                    <div className="p-3 rounded-lg bg-gray-50 border border-gray-100">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-[10px] font-bold text-purple-600 bg-purple-100 px-1.5 py-0.5 rounded">AI</span>
                        <span className="text-xs text-gray-500">Combination Mode</span>
                      </div>
                      <p className="text-sm font-semibold">Stackable</p>
                    </div>

                    {/* Codes & Limits */}
                    <div className="grid grid-cols-2 gap-3">
                      <div className="p-3 rounded-lg bg-gray-50 border border-gray-100">
                        <p className="text-xs text-gray-500 mb-1">Codes</p>
                        <p className="text-sm font-medium">No code required</p>
                      </div>
                      <div className="p-3 rounded-lg bg-gray-50 border border-gray-100">
                        <p className="text-xs text-gray-500 mb-1">Limits</p>
                        <p className="text-sm font-medium">No limits set</p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Validation */}
                <div className="card p-5">
                  <h3 className="text-sm font-semibold text-gray-900 mb-3">Validation</h3>
                  <div className="space-y-2">
                    {[
                      { check: "Priority within 0–1000", passed: true },
                      { check: "End date after start date", passed: true },
                      { check: "Percent between 1–100", passed: true },
                      { check: "At least one site selected", passed: true },
                      { check: "Template matches reward type", passed: true },
                    ].map((v, idx) => (
                      <div key={idx} className="flex items-center gap-2 text-sm">
                        <CheckCircle2 size={14} className="text-emerald-500" />
                        <span className="text-gray-700">{v.check}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            )}

            {!hasResult && !clarifying && (
              <div className="card p-16 text-center">
                <Sparkles size={40} className="text-purple-200 mx-auto mb-3" />
                <p className="text-lg font-semibold text-gray-400">Describe your promotion</p>
                <p className="text-sm text-gray-300 mt-1">
                  Write a plain-language brief and the AI will generate a structured draft
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </AppShell>
  );
}
